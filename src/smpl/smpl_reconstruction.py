#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from SMPL_reconstruction.smpl_reconstruct import (
    connected_bone_roll_deltas,
    connected_bone_roll_score,
    estimate_pose_from_joints,
    major_bone_position_score,
    mpjpe,
    resolve_smplh_model_npz,
    smplh_vertices_joints_blender,
    unproject_uvz_to_world,
    write_obj,
)


def _roll_delta_rms_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    keep = int(min(a.shape[0], b.shape[0]))
    if keep == 0:
        return 0.0
    delta = a[:keep] - b[:keep]
    return float(np.degrees(np.sqrt(np.mean(delta * delta))))


def _roll_score(
    *,
    pose_axis_angle: np.ndarray,
    posed_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents_22: np.ndarray,
    full_joint_count: int,
) -> float:
    return connected_bone_roll_score(
        pose_axis_angle=pose_axis_angle,
        posed_joints_blender=posed_joints[:22],
        rest_joints_blender=rest_joints[:22],
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )


def _roll_deltas(
    *,
    pose_axis_angle: np.ndarray,
    posed_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents_22: np.ndarray,
    full_joint_count: int,
) -> np.ndarray:
    return connected_bone_roll_deltas(
        pose_axis_angle=pose_axis_angle,
        posed_joints_blender=posed_joints[:22],
        rest_joints_blender=rest_joints[:22],
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )


def _render_scores(
    *,
    pose_axis_angle: np.ndarray,
    posed_joints: np.ndarray,
    rest_joints: np.ndarray,
    gt_roll_deltas: np.ndarray,
    gt_joints: np.ndarray,
    target_joints_world: np.ndarray,
    parents_22: np.ndarray,
    full_joint_count: int,
) -> dict[str, float]:
    roll_deltas = _roll_deltas(
        pose_axis_angle=pose_axis_angle,
        posed_joints=posed_joints,
        rest_joints=rest_joints,
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )
    return {
        "roll_from_rest_deg": _roll_delta_rms_deg(roll_deltas, np.zeros_like(roll_deltas)),
        "roll_from_gt_deg": _roll_delta_rms_deg(roll_deltas, gt_roll_deltas),
        "pos_from_gt": major_bone_position_score(posed_joints[:22], gt_joints[:22], parents_22),
        "pos_from_3d_joints": major_bone_position_score(posed_joints[:22], target_joints_world[:22], parents_22),
    }


def _overlay_scores_on_png(path: Path, scores: dict[str, float]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        print(f"Pillow is not installed; skipping score overlay for {path}")
        return

    if not path.exists():
        return

    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, image.width // 34))
    except Exception:
        font = ImageFont.load_default()

    lines = [
        f"roll/rest: {scores['roll_from_rest_deg']:.2f} deg",
        f"roll/GT:   {scores['roll_from_gt_deg']:.2f} deg",
        f"pos/GT:    {scores['pos_from_gt']:.4f}",
        f"pos/3D:    {scores['pos_from_3d_joints']:.4f}",
    ]
    pad = max(10, image.width // 64)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max(box[2] - box[0] for box in line_boxes)
    text_h = sum(box[3] - box[1] for box in line_boxes) + pad // 2 * (len(lines) - 1)
    box = (pad, pad, pad * 2 + text_w, pad * 2 + text_h)
    draw.rounded_rectangle(box, radius=6, fill=(0, 0, 0, 168))

    y = pad + pad // 2
    for line, bbox in zip(lines, line_boxes):
        draw.text((pad + pad // 2, y), line, font=font, fill=(255, 255, 255, 255))
        y += (bbox[3] - bbox[1]) + pad // 2

    image.save(path)


def _overlay_render_scores(render_dir: Path, per_render_scores: dict[str, dict[str, float]]) -> None:
    render_files = {
        "gt": "gt.png",
        "recon_betas": "recon_betas.png",
        "recon_zero_betas": "recon_zero_betas.png",
    }
    for key, filename in render_files.items():
        if key in per_render_scores:
            _overlay_scores_on_png(render_dir / filename, per_render_scores[key])


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruct SMPL-H pose from 22 bones using local bone-angle reconstruction.")
    p.add_argument("--meta", required=True, help="Path to dataset JSON produced by create_poses.py")
    p.add_argument("--out-dir", default="SMPL_reconstruction/output", help="Output directory")
    p.add_argument("--render", action="store_true", help="Render with Blender if available")
    p.add_argument("--blender", default="blender", help="Blender executable")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--engine", default="BLENDER_EEVEE", choices=("BLENDER_EEVEE", "CYCLES"))
    p.add_argument("--samples", type=int, default=16)
    args = p.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())

    joints_uvz = np.asarray(meta["joints_3d"], dtype=np.float64)
    parents_22 = np.asarray(meta["parents"], dtype=np.int64)
    gender = str(meta.get("smpl", {}).get("gender", "neutral"))
    center_xy = tuple(float(x) for x in meta.get("smpl", {}).get("center_xy", [0.0, 0.0]))
    z_min = float(meta.get("smpl", {}).get("z_min", 0.0))
    betas = np.asarray(meta.get("smpl", {}).get("betas", []), dtype=np.float64)
    gt_pose = np.asarray(meta.get("smpl", {}).get("pose_axis_angle", []), dtype=np.float64)
    zero_betas = np.zeros_like(betas)

    model_npz = resolve_smplh_model_npz(gender=gender)
    target_joints_world = unproject_uvz_to_world(joints_uvz, meta)

    gt_vertices, gt_joints, _p, faces = smplh_vertices_joints_blender(
        model_npz_path=model_npz,
        pose_axis_angle=gt_pose,
        betas=betas,
        center_xy=center_xy,
        z_min=z_min,
    )

    zero_pose = np.zeros_like(gt_pose)
    _rest_v_betas, rest_joints_betas, _parents_betas, _faces = smplh_vertices_joints_blender(
        model_npz_path=model_npz,
        pose_axis_angle=zero_pose,
        betas=betas,
        center_xy=center_xy,
        z_min=z_min,
    )
    _rest_v_zero, rest_joints_zero, _parents_zero, _faces = smplh_vertices_joints_blender(
        model_npz_path=model_npz,
        pose_axis_angle=zero_pose,
        betas=zero_betas,
        center_xy=center_xy,
        z_min=z_min,
    )
    full_joint_count = int(rest_joints_betas.shape[0])

    recon_betas_pose = estimate_pose_from_joints(
        target_joints_blender=target_joints_world,
        rest_joints_blender=rest_joints_betas[:22],
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )
    recon_zero_betas_pose = estimate_pose_from_joints(
        target_joints_blender=target_joints_world,
        rest_joints_blender=rest_joints_zero[:22],
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )

    recon_betas_vertices, recon_betas_joints, _p2, _faces2 = smplh_vertices_joints_blender(
        model_npz_path=model_npz,
        pose_axis_angle=recon_betas_pose,
        betas=betas,
        center_xy=center_xy,
        z_min=z_min,
    )
    recon_zero_betas_vertices, recon_zero_betas_joints, _p3, _faces3 = smplh_vertices_joints_blender(
        model_npz_path=model_npz,
        pose_axis_angle=recon_zero_betas_pose,
        betas=zero_betas,
        center_xy=center_xy,
        z_min=z_min,
    )

    gt_roll_deltas = _roll_deltas(
        pose_axis_angle=gt_pose,
        posed_joints=gt_joints,
        rest_joints=rest_joints_betas,
        parents_22=parents_22,
        full_joint_count=full_joint_count,
    )
    per_render_scores = {
        "gt": _render_scores(
            pose_axis_angle=gt_pose,
            posed_joints=gt_joints,
            rest_joints=rest_joints_betas,
            gt_roll_deltas=gt_roll_deltas,
            gt_joints=gt_joints,
            target_joints_world=target_joints_world,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "recon_betas": _render_scores(
            pose_axis_angle=recon_betas_pose,
            posed_joints=recon_betas_joints,
            rest_joints=rest_joints_betas,
            gt_roll_deltas=gt_roll_deltas,
            gt_joints=gt_joints,
            target_joints_world=target_joints_world,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "recon_zero_betas": _render_scores(
            pose_axis_angle=recon_zero_betas_pose,
            posed_joints=recon_zero_betas_joints,
            rest_joints=rest_joints_zero,
            gt_roll_deltas=gt_roll_deltas,
            gt_joints=gt_joints,
            target_joints_world=target_joints_world,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_obj = out_dir / "gt.obj"
    recon_betas_obj = out_dir / "recon_betas.obj"
    recon_zero_betas_obj = out_dir / "recon_zero_betas.obj"
    write_obj(gt_obj, gt_vertices, faces)
    write_obj(recon_betas_obj, recon_betas_vertices, faces)
    write_obj(recon_zero_betas_obj, recon_zero_betas_vertices, faces)

    scores = {
        "gt_vs_unproject_mpjpe_22": mpjpe(gt_joints[:22], target_joints_world[:22]),
        "recon_betas_vs_target_mpjpe_22": mpjpe(recon_betas_joints[:22], target_joints_world[:22]),
        "recon_zero_betas_vs_target_mpjpe_22": mpjpe(recon_zero_betas_joints[:22], target_joints_world[:22]),
        "input_position_score_to_gt": major_bone_position_score(target_joints_world[:22], gt_joints[:22], parents_22),
        "gt_position_score_to_gt": major_bone_position_score(gt_joints[:22], gt_joints[:22], parents_22),
        "recon_betas_position_score_to_gt": major_bone_position_score(recon_betas_joints[:22], gt_joints[:22], parents_22),
        "recon_zero_betas_position_score_to_gt": major_bone_position_score(
            recon_zero_betas_joints[:22], gt_joints[:22], parents_22
        ),
        "recon_betas_position_score_to_input": major_bone_position_score(
            recon_betas_joints[:22], target_joints_world[:22], parents_22
        ),
        "recon_zero_betas_position_score_to_input": major_bone_position_score(
            recon_zero_betas_joints[:22], target_joints_world[:22], parents_22
        ),
        "gt_roll_score_deg": _roll_score(
            pose_axis_angle=gt_pose,
            posed_joints=gt_joints,
            rest_joints=rest_joints_betas,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "recon_betas_roll_score_deg": _roll_score(
            pose_axis_angle=recon_betas_pose,
            posed_joints=recon_betas_joints,
            rest_joints=rest_joints_betas,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "recon_zero_betas_roll_score_deg": _roll_score(
            pose_axis_angle=recon_zero_betas_pose,
            posed_joints=recon_zero_betas_joints,
            rest_joints=rest_joints_zero,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "rest_roll_score_deg": _roll_score(
            pose_axis_angle=np.zeros((full_joint_count * 3,), dtype=np.float64),
            posed_joints=rest_joints_betas,
            rest_joints=rest_joints_betas,
            parents_22=parents_22,
            full_joint_count=full_joint_count,
        ),
        "per_render": per_render_scores,
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print("MPJPE vs input bones, first 22 (GT / recon betas / recon zero betas):")
    print(scores["gt_vs_unproject_mpjpe_22"], scores["recon_betas_vs_target_mpjpe_22"], scores["recon_zero_betas_vs_target_mpjpe_22"])
    print("Position score vs GT, 22 constrained major bones (input / GT / recon betas / recon zero betas):")
    print(
        scores["input_position_score_to_gt"],
        scores["gt_position_score_to_gt"],
        scores["recon_betas_position_score_to_gt"],
        scores["recon_zero_betas_position_score_to_gt"],
    )
    print("Position score vs input bones, 22 constrained major bones (recon betas / recon zero betas):")
    print(scores["recon_betas_position_score_to_input"], scores["recon_zero_betas_position_score_to_input"])
    print("Roll distance from rest pose RMS deg (GT / recon betas / recon zero betas / rest):")
    print(
        scores["gt_roll_score_deg"],
        scores["recon_betas_roll_score_deg"],
        scores["recon_zero_betas_roll_score_deg"],
        scores["rest_roll_score_deg"],
    )
    print("Wrote:", gt_obj)
    print("Wrote:", recon_betas_obj)
    print("Wrote:", recon_zero_betas_obj)
    print("Wrote:", out_dir / "scores.json")

    if args.render:
        blender_exe = args.blender
        if shutil.which(blender_exe) is None:
            raise FileNotFoundError(f"Blender not found: {blender_exe}")
        render_dir = out_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            blender_exe,
            "-b",
            "--python",
            str((Path(__file__).resolve().parent / "blender_render_compare.py")),
            "--",
            "--meta",
            str(meta_path.resolve()),
            "--gt-obj",
            str(gt_obj.resolve()),
            "--recon-betas-obj",
            str(recon_betas_obj.resolve()),
            "--recon-zero-betas-obj",
            str(recon_zero_betas_obj.resolve()),
            "--out-dir",
            str(render_dir.resolve()),
            "--resolution",
            str(int(args.resolution)),
            "--engine",
            str(args.engine),
            "--samples",
            str(int(args.samples)),
        ]
        run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(run.stdout)
        if run.returncode != 0:
            raise RuntimeError(f"Blender render failed with code {run.returncode}")
        _overlay_render_scores(render_dir, per_render_scores)
        print("Renders:", render_dir)


if __name__ == "__main__":
    main()
