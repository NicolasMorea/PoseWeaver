#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a PoseWeaver SMPL pose against a Sketch2UV mask contour")
    parser.add_argument("--pose-npz", required=True)
    parser.add_argument("--uv-json", required=True)
    parser.add_argument("--smpl-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--debug-dir", default="renders")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--render-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--pose-reg", type=float, default=0.018)
    parser.add_argument("--beta-reg", type=float, default=0.030)
    parser.add_argument("--samples", type=int, default=70000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--camera-eye", nargs=3, type=float, required=True)
    parser.add_argument("--camera-direction", nargs=3, type=float, required=True)
    parser.add_argument("--camera-up", nargs=3, type=float, required=True)
    parser.add_argument("--camera-fov-deg", type=float, default=0.7348438501358032 * 180.0 / math.pi)
    parser.add_argument("--raw-rig-root", nargs=3, type=float)
    parser.add_argument("--raw-rig-height", type=float)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_escape_path(path: Path) -> str:
    return str(path.resolve())


def resolve_from_json(base: Path, value: str, fallback: Path) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else base / path
    return fallback


def load_uv_mask(uv_json_path: Path):
    import numpy as np

    uv_meta = read_json(uv_json_path)
    uv_npz_path = resolve_from_json(
        uv_json_path.parent,
        str(uv_meta.get("uv_npz_path", "")),
        uv_json_path.with_suffix("").parent / (uv_json_path.with_suffix("").name + "_uv.npz"),
    )
    uv_npz = np.load(uv_npz_path)
    mask = uv_npz["mask"].astype("float32")
    return mask, uv_npz_path, uv_meta


def load_pose(path: Path):
    import numpy as np

    z = np.load(path)
    pose = z["poses"].astype("float32").reshape(-1)
    betas = z["betas"].astype("float32").reshape(-1) if "betas" in z.files else np.zeros((16,), dtype=np.float32)
    mirror_x = bool(int(np.asarray(z["poseweaver_mirror_x"]).reshape(-1)[0])) if "poseweaver_mirror_x" in z.files else False
    no_root_fix = bool(int(np.asarray(z["poseweaver_no_root_fix"]).reshape(-1)[0])) if "poseweaver_no_root_fix" in z.files else True
    gender_index = int(np.asarray(z["gender_index"]).reshape(-1)[0]) if "gender_index" in z.files else 0
    gender = "male"
    if "gender" in z.files:
        raw = z["gender"]
        try:
            gender = bytes(raw.tolist()).decode("utf-8")
        except Exception:
            gender = "female" if gender_index == 1 else "neutral" if gender_index == 2 else "male"
    return pose, betas, mirror_x, no_root_fix, gender, gender_index


def parents_from_kintree(kintree):
    import numpy as np

    k = np.asarray(kintree)
    joint_count = int(k.shape[1])
    row0 = k[0].astype("int64")
    row1 = k[1].astype("int64")

    def is_ids(row) -> bool:
        return all(int(row[i]) == i for i in range(joint_count))

    if is_ids(row0) and not is_ids(row1):
        ids, parents_raw = row0, row1
    elif is_ids(row1) and not is_ids(row0):
        ids, parents_raw = row1, row0
    elif int(row0[0]) in (-1, 4294967295) and int(row1[0]) not in (-1, 4294967295):
        ids, parents_raw = row1, row0
    else:
        ids, parents_raw = row0, row1

    id_to_col = {int(v): i for i, v in enumerate(ids.tolist())}
    parents = []
    for pid in parents_raw.tolist():
        pid = int(pid)
        parents.append(-1 if pid in (-1, 4294967295) else int(id_to_col.get(pid, -1)))
    return parents


def load_smpl_model(path: Path, device):
    import numpy as np
    import torch

    z = np.load(path)
    shapedirs = z["shapedirs"].astype("float32")
    if shapedirs.ndim == 3:
        shapedirs = shapedirs.reshape((-1, shapedirs.shape[-1]))
    posedirs = z["posedirs"].astype("float32")
    if posedirs.ndim == 3:
        posedirs = posedirs.reshape((-1, posedirs.shape[-1]))

    faces = z["f"].astype("int64") if "f" in z.files else z["F"].astype("int64")
    return {
        "v_template": torch.as_tensor(z["v_template"].astype("float32"), device=device),
        "shapedirs": torch.as_tensor(shapedirs, device=device),
        "posedirs": torch.as_tensor(posedirs, device=device),
        "J_regressor": torch.as_tensor(z["J_regressor"].astype("float32"), device=device),
        "weights": torch.as_tensor(z["weights"].astype("float32"), device=device),
        "faces": torch.as_tensor(faces, dtype=torch.long, device=device),
        "parents": parents_from_kintree(z["kintree_table"]),
    }


def rodrigues(axis_angle):
    import torch

    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / theta
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(K)
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    return eye + sin * K + (1.0 - cos) * (K @ K)


def smpl_lbs(model: dict[str, Any], pose_axis_angle, betas):
    import torch

    v_template = model["v_template"]
    vertex_count = int(v_template.shape[0])
    joint_count = int(model["weights"].shape[1])
    beta_count = int(model["shapedirs"].shape[1])

    betas = betas[:beta_count]
    if betas.shape[0] < beta_count:
        betas = torch.nn.functional.pad(betas, (0, beta_count - betas.shape[0]))

    pose = pose_axis_angle[: joint_count * 3]
    if pose.shape[0] < joint_count * 3:
        pose = torch.nn.functional.pad(pose, (0, joint_count * 3 - pose.shape[0]))
    pose = pose.reshape(joint_count, 3)

    v_shaped_flat = v_template.reshape(-1) + model["shapedirs"] @ betas
    v_shaped = v_shaped_flat.reshape(vertex_count, 3)
    joints = model["J_regressor"] @ v_shaped

    R = rodrigues(pose)
    ident = torch.eye(3, dtype=v_template.dtype, device=v_template.device)
    lrotmin = (R[1:] - ident).reshape(-1)
    v_posed = (v_shaped_flat + model["posedirs"] @ lrotmin).reshape(vertex_count, 3)

    transforms = []
    for j, parent in enumerate(model["parents"]):
        A = torch.eye(4, dtype=v_template.dtype, device=v_template.device)
        A[:3, :3] = R[j]
        if parent < 0:
            A[:3, 3] = joints[j]
            transforms.append(A)
        else:
            A[:3, 3] = joints[j] - joints[parent]
            transforms.append(transforms[parent] @ A)

    rest_transforms = []
    for j, G in enumerate(transforms):
        rest = torch.eye(4, dtype=v_template.dtype, device=v_template.device)
        rest[:3, 3] = -joints[j]
        rest_transforms.append(G @ rest)

    T = torch.stack(rest_transforms, dim=0)
    T_vertex = torch.einsum("vj,jab->vab", model["weights"], T)
    homo = torch.cat([v_posed, torch.ones((vertex_count, 1), dtype=v_template.dtype, device=v_template.device)], dim=1)
    return (T_vertex @ homo[:, :, None])[:, :3, 0]


def make_target_tensors(mask_np, size: int, device):
    import torch
    import torch.nn.functional as F

    mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device)[None, None]
    mask = F.interpolate(mask, size=(size, size), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    mask = (mask > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    eroded = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
    contour = (dilated - eroded).clamp(0.0, 1.0)
    ys, xs = torch.where(mask[0, 0] > 0.5)
    if xs.numel() == 0:
        raise RuntimeError("Sketch2UV mask is empty")
    bbox = (
        float(xs.float().min()),
        float(ys.float().min()),
        float(xs.float().max()),
        float(ys.float().max()),
    )
    return mask, contour, bbox


def sample_surface_points(faces, count: int, device):
    import torch

    face_count = int(faces.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    face_ids = torch.randint(0, face_count, (count,), generator=generator, dtype=torch.long).to(device)
    uv = torch.rand((count, 2), generator=generator, dtype=torch.float32).to(device)
    su = torch.sqrt(uv[:, 0].clamp_min(1e-8))
    w0 = 1.0 - su
    w1 = su * (1.0 - uv[:, 1])
    w2 = su * uv[:, 1]
    bary = torch.stack([w0, w1, w2], dim=1)
    tri = faces[face_ids]
    return tri, bary


def align_vertices_to_scene(vertices, model: dict[str, Any], mirror_x: bool, raw_rig_root, raw_rig_height):
    import torch

    verts = vertices
    if mirror_x:
        verts = verts.clone()
        verts[:, 0] *= -1.0

    if raw_rig_root is not None:
        joints = model["J_regressor"] @ verts
        current_root = joints[0]
        scale = torch.tensor(1.0, dtype=vertices.dtype, device=vertices.device)
        if raw_rig_height is not None and float(raw_rig_height) > 1e-6:
            smpl_height = (joints[:, 1].max() - joints[:, 1].min()).clamp_min(1e-6)
            scale = torch.clamp(
                torch.tensor(float(raw_rig_height), dtype=vertices.dtype, device=vertices.device) / smpl_height,
                0.05,
                20.0,
            )
        root = torch.tensor(raw_rig_root, dtype=vertices.dtype, device=vertices.device)
        verts = (verts - current_root[None, :]) * scale + root[None, :]

    return verts


def make_camera(camera_eye, camera_direction, camera_up, fov_deg: float, dtype, device):
    import torch

    eye = torch.tensor(camera_eye, dtype=dtype, device=device)
    forward = torch.tensor(camera_direction, dtype=dtype, device=device)
    up = torch.tensor(camera_up, dtype=dtype, device=device)
    forward = forward / torch.linalg.norm(forward).clamp_min(1e-8)
    up = up - forward * torch.dot(forward, up)
    up = up / torch.linalg.norm(up).clamp_min(1e-8)
    right = torch.cross(forward, up, dim=0)
    right = right / torch.linalg.norm(right).clamp_min(1e-8)
    true_up = torch.cross(right, forward, dim=0)
    tan_half_fov = math.tan(math.radians(float(fov_deg)) * 0.5)
    return eye, forward, right, true_up, tan_half_fov


def render_soft_silhouette(vertices, faces, sample_tri, sample_bary, camera, size: int):
    import torch

    eye, forward, right, up, tan_half_fov = camera
    tri, bary = sample_tri, sample_bary
    pts = (
        vertices[tri[:, 0]] * bary[:, 0:1]
        + vertices[tri[:, 1]] * bary[:, 1:2]
        + vertices[tri[:, 2]] * bary[:, 2:3]
    )
    rel = pts - eye[None, :]
    depth = rel @ forward
    x_cam = rel @ right
    y_cam = rel @ up

    valid_depth = depth > 1e-4
    inv = 1.0 / depth.clamp_min(1e-4)
    ndc_x = x_cam * inv / tan_half_fov
    ndc_y = y_cam * inv / tan_half_fov
    px = (ndc_x * 0.5 + 0.5) * float(size)
    py = (-ndc_y * 0.5 + 0.5) * float(size)

    x0 = torch.floor(px).long()
    y0 = torch.floor(py).long()
    dx = px - x0.float()
    dy = py - y0.float()

    density = torch.zeros((size * size,), dtype=vertices.dtype, device=vertices.device)
    for ox, wx in ((0, 1.0 - dx), (1, dx)):
        xi = x0 + ox
        for oy, wy in ((0, 1.0 - dy), (1, dy)):
            yi = y0 + oy
            valid = valid_depth & (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
            if valid.any():
                idx = yi[valid] * size + xi[valid]
                density.scatter_add_(0, idx, (wx[valid] * wy[valid]).clamp_min(0.0))
    density = density.reshape(1, 1, size, size)
    return 1.0 - torch.exp(-density * 0.55)


def contour_from_soft_mask(mask):
    import torch.nn.functional as F

    dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    eroded = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
    return (dilated - eroded).clamp(0.0, 1.0)


def save_gray(path: Path, tensor) -> None:
    import numpy as np
    from PIL import Image

    arr = tensor.detach().float().cpu().numpy()
    arr = np.squeeze(arr)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    value = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 0] = value
    rgba[..., 1] = value
    rgba[..., 2] = value
    rgba[..., 3] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_overlay(path: Path, target_mask, render_mask, target_contour, render_contour) -> None:
    import numpy as np
    from PIL import Image

    tgt = np.squeeze(target_mask.detach().cpu().numpy())
    rnd = np.squeeze(render_mask.detach().cpu().numpy())
    tc = np.squeeze(target_contour.detach().cpu().numpy())
    rc = np.squeeze(render_contour.detach().cpu().numpy())
    h, w = tgt.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(tc * 255.0 + np.maximum(tgt - rnd, 0.0) * 90.0, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(rc * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(np.minimum(tgt, rnd) * 80.0, 0, 255).astype(np.uint8)
    alpha = np.clip((tc + rc + np.abs(tgt - rnd) * 0.5) * 255.0, 0, 255)
    rgba[..., 3] = alpha.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_npz(path: Path, pose, betas, gender: str, gender_index: int, mirror_x: bool, no_root_fix: bool) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    np.savez_compressed(
        path,
        poses=np.asarray(pose, dtype=np.float64).reshape(1, -1),
        betas=np.asarray(betas, dtype=np.float64).reshape(-1),
        gender_index=np.asarray([gender_index], dtype=np.int32),
        poseweaver_no_root_fix=np.asarray([1 if no_root_fix else 0], dtype=np.int32),
        poseweaver_mirror_x=np.asarray([1 if mirror_x else 0], dtype=np.int32),
        gender=np.frombuffer(gender.encode("utf-8"), dtype=np.uint8),
    )


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )

    pose0_np, betas0_np, mirror_x, no_root_fix, gender, gender_index = load_pose(Path(args.pose_npz))
    mask_np, uv_npz_path, uv_meta = load_uv_mask(Path(args.uv_json))
    size = int(max(64, min(384, args.render_size)))
    iterations = int(max(1, args.iterations))

    model = load_smpl_model(Path(args.smpl_model), device)
    pose0 = torch.as_tensor(pose0_np, dtype=torch.float32, device=device)
    betas0 = torch.as_tensor(betas0_np, dtype=torch.float32, device=device)
    target_mask, target_contour, target_bbox = make_target_tensors(mask_np, size, device)
    sample_tri, sample_bary = sample_surface_points(model["faces"], int(max(1000, args.samples)), device)
    raw_rig_root = args.raw_rig_root
    raw_rig_height = args.raw_rig_height
    camera = make_camera(
        args.camera_eye,
        args.camera_direction,
        args.camera_up,
        args.camera_fov_deg,
        torch.float32,
        device,
    )

    with torch.no_grad():
        verts0 = align_vertices_to_scene(
            smpl_lbs(model, pose0, betas0),
            model,
            mirror_x,
            raw_rig_root,
            raw_rig_height,
        )
        initial_render = render_soft_silhouette(verts0, model["faces"], sample_tri, sample_bary, camera, size)
        initial_contour = contour_from_soft_mask(initial_render)
        initial_mask_loss = F.binary_cross_entropy(initial_render.clamp(1e-5, 1.0 - 1e-5), target_mask)
        initial_contour_loss = F.l1_loss(initial_contour, target_contour)
        initial_loss_value = float((initial_mask_loss + 2.4 * initial_contour_loss).cpu())

    body_joint_count = min(22, int(model["weights"].shape[1]))
    pose_delta = torch.zeros((body_joint_count * 3,), dtype=torch.float32, device=device, requires_grad=True)
    betas_delta = torch.zeros_like(betas0, requires_grad=True)

    optimizer = torch.optim.Adam([pose_delta, betas_delta], lr=float(args.lr))
    history: list[dict[str, float]] = []
    initial_loss = initial_loss_value

    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        pose = pose0.clone()
        pose[: body_joint_count * 3] = pose[: body_joint_count * 3] + pose_delta
        betas = betas0 + betas_delta
        verts = align_vertices_to_scene(
            smpl_lbs(model, pose, betas),
            model,
            mirror_x,
            raw_rig_root,
            raw_rig_height,
        )
        render_mask = render_soft_silhouette(verts, model["faces"], sample_tri, sample_bary, camera, size)
        render_contour = contour_from_soft_mask(render_mask)

        mask_loss = F.binary_cross_entropy(render_mask.clamp(1e-5, 1.0 - 1e-5), target_mask)
        contour_loss = F.l1_loss(render_contour, target_contour)
        pose_reg = torch.mean(pose_delta * pose_delta)
        beta_reg = torch.mean(betas_delta * betas_delta)
        loss = mask_loss + 2.4 * contour_loss + float(args.pose_reg) * pose_reg + float(args.beta_reg) * beta_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([pose_delta, betas_delta], 1.0)
        optimizer.step()

        with torch.no_grad():
            pose_delta.clamp_(-0.65, 0.65)
            betas_delta.clamp_(-0.75, 0.75)
        if step == iterations - 1 or step % max(1, iterations // 10) == 0:
            history.append(
                {
                    "iteration": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "mask_loss": float(mask_loss.detach().cpu()),
                    "contour_loss": float(contour_loss.detach().cpu()),
                }
            )

    with torch.no_grad():
        final_pose = pose0.clone()
        final_pose[: body_joint_count * 3] += pose_delta
        final_betas = betas0 + betas_delta
        initial_render = render_soft_silhouette(verts0, model["faces"], sample_tri, sample_bary, camera, size)
        final_vertices = align_vertices_to_scene(
            smpl_lbs(model, final_pose, final_betas),
            model,
            mirror_x,
            raw_rig_root,
            raw_rig_height,
        )
        final_render = render_soft_silhouette(final_vertices, model["faces"], sample_tri, sample_bary, camera, size)
        final_contour = contour_from_soft_mask(final_render)
        final_mask_loss = F.binary_cross_entropy(final_render.clamp(1e-5, 1.0 - 1e-5), target_mask)
        final_contour_loss = F.l1_loss(final_contour, target_contour)
        final_loss = float((final_mask_loss + 2.4 * final_contour_loss).cpu())

    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = "rig_optimization"
    target_contour_path = debug_dir / f"{stem}_target_contour.png"
    initial_render_path = debug_dir / f"{stem}_initial_silhouette.png"
    optimized_render_path = debug_dir / f"{stem}_optimized_silhouette.png"
    loss_overlay_path = debug_dir / f"{stem}_contour_loss_overlay.png"

    save_gray(target_contour_path, target_contour)
    save_gray(initial_render_path, initial_render)
    save_gray(optimized_render_path, final_render)
    save_overlay(loss_overlay_path, target_mask, final_render, target_contour, final_contour)
    save_npz(output_npz, final_pose.cpu().numpy(), final_betas.cpu().numpy(), gender, gender_index, mirror_x, no_root_fix)

    output = {
        "schema": "poseweaver.rig_optimization.v1",
        "method": "torch_smpl_lbs_with_differentiable_soft_silhouette_splatting",
        "differentiable_rendering": True,
        "source_pose_npz": json_escape_path(Path(args.pose_npz)),
        "source_uv_json": json_escape_path(Path(args.uv_json)),
        "source_uv_npz": json_escape_path(uv_npz_path),
        "source_sketch": uv_meta.get("source_sketch", ""),
        "smpl_model": json_escape_path(Path(args.smpl_model)),
        "optimized_pose_npz": json_escape_path(output_npz),
        "gender": gender,
        "gender_index": gender_index,
        "mirror_smpl_x": mirror_x,
        "parameters": {
            "iterations": iterations,
            "render_size": size,
            "learning_rate": float(args.lr),
            "pose_regularization": float(args.pose_reg),
            "beta_regularization": float(args.beta_reg),
            "surface_samples": int(max(1000, args.samples)),
            "optimized_body_joints": body_joint_count,
            "camera_model": "poseweaver_scene_view_perspective",
            "camera_eye": [float(v) for v in args.camera_eye],
            "camera_direction": [float(v) for v in args.camera_direction],
            "camera_up": [float(v) for v in args.camera_up],
            "camera_fov_degrees": float(args.camera_fov_deg),
            "raw_rig_root": [float(v) for v in raw_rig_root] if raw_rig_root is not None else None,
            "raw_rig_height": float(raw_rig_height) if raw_rig_height is not None else None,
            "device": str(device),
        },
        "loss": {
            "initial": float(initial_loss if initial_loss is not None else final_loss),
            "final": final_loss,
            "final_mask": float(final_mask_loss.cpu()),
            "final_contour": float(final_contour_loss.cpu()),
            "history": history,
        },
        "betas": [float(v) for v in final_betas.detach().cpu().numpy().tolist()],
        "pose_axis_angle": [float(v) for v in final_pose.detach().cpu().numpy().tolist()],
        "debug_images": {
            "target_contour_path": json_escape_path(target_contour_path),
            "initial_silhouette_path": json_escape_path(initial_render_path),
            "optimized_silhouette_path": json_escape_path(optimized_render_path),
            "contour_loss_overlay_path": json_escape_path(loss_overlay_path),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_json.with_suffix(output_json.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp.replace(output_json)
    print(json.dumps({"ok": True, "output_json": str(output_json), "final_loss": final_loss}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr, flush=True)
        raise
