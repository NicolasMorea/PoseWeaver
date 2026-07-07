#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def emit(payload: dict[str, Any]) -> None:
    print("POSEWEAVER_JSON " + json.dumps(payload, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent VitPose inference worker")
    parser.add_argument("--checkpoint", required=True, help="Path to the VitPose checkpoint")
    parser.add_argument(
        "--vitpose-repo",
        default=os.environ.get("POSEWEAVER_VITPOSE_REPO", ""),
        help="Path to the sketch_to_pose repository containing ViTPose/ and mmcv/",
    )
    parser.add_argument(
        "--pose-config",
        default="ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py",
        help="Absolute path or path relative to --vitpose-repo",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    return parser.parse_args()


def find_vitpose_repo(explicit_path: str) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    poseweaver_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        (
            poseweaver_root.parent / "sketch_to_pose",
            Path.cwd().parent / "sketch_to_pose",
        )
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "ViTPose" / "mmpose").is_dir() and (candidate / "mmcv" / "mmcv").is_dir():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the sketch_to_pose repository containing ViTPose/ and mmcv/. "
        f"Searched: {searched}. Set POSEWEAVER_VITPOSE_REPO to override."
    )


def add_local_import_paths(repo_root: Path) -> None:
    # Keep the source versions used by the reference inference script ahead of
    # any globally installed mmpose/mmcv packages.
    sys.path.insert(0, str(repo_root / "ViTPose"))
    sys.path.insert(0, str(repo_root / "mmcv"))


def resolve_config(repo_root: Path, config_arg: str) -> Path:
    config = Path(config_arg).expanduser()
    if not config.is_absolute():
        config = repo_root / config
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(f"VitPose config not found: {config}")
    return config


def load_model(checkpoint_path: Path, repo_root: Path, pose_config: Path, device_name: str):
    add_local_import_paths(repo_root)

    import torch
    from mmpose.apis import init_pose_model
    from mmpose.datasets import DatasetInfo

    if device_name == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda":
        device = "cuda:0"
    else:
        device = device_name

    pose_model = init_pose_model(str(pose_config), str(checkpoint_path), device=device)
    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg is not None else None
    return pose_model, dataset, dataset_info, device


def write_prediction(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    device: str,
    image_width: int,
    image_height: int,
    raw_keypoints: list[list[float]],
) -> None:
    keypoints = []
    for index, values in enumerate(raw_keypoints):
        if len(values) < 3:
            continue
        keypoints.append(
            {
                "index": index,
                "name": COCO_KEYPOINT_NAMES[index] if index < len(COCO_KEYPOINT_NAMES) else f"keypoint_{index}",
                "position": [float(values[0]), float(values[1])],
                "score": float(values[2]),
            }
        )

    result = {
        "schema": "poseweaver.vitpose.v1",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": [image_width, image_height],
        "keypoint_format": "coco17",
        "keypoints": keypoints,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def infer_one(
    pose_model,
    dataset: str,
    dataset_info,
    checkpoint_path: Path,
    device: str,
    command: dict[str, Any],
) -> None:
    from PIL import Image
    from mmpose.apis import inference_top_down_pose_model

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])

    emit({"event": "busy", "request_id": request_id, "message": "Running VitPose inference"})
    with Image.open(sketch_path) as image:
        image_width, image_height = image.size

    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        str(sketch_path),
        person_results=None,
        bbox_thr=None,
        format="xywh",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
    )

    raw_keypoints = pose_results[0]["keypoints"].tolist() if pose_results else []
    write_prediction(
        output_path,
        sketch_path,
        checkpoint_path,
        device,
        image_width,
        image_height,
        raw_keypoints,
    )
    emit(
        {
            "event": "result",
            "request_id": request_id,
            "ok": True,
            "message": f"2D pose predicted ({len(raw_keypoints)} keypoints)",
            "output_path": str(output_path),
        }
    )


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    try:
        repo_root = find_vitpose_repo(args.vitpose_repo)
        pose_config = resolve_config(repo_root, args.pose_config)
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        pose_model, dataset, dataset_info, device = load_model(
            checkpoint_path,
            repo_root,
            pose_config,
            args.device,
        )
        emit({"event": "ready", "message": f"Model loaded on {device}"})
    except Exception as exc:
        emit({"event": "error", "message": f"Failed to load model: {exc}"})
        traceback.print_exc()
        return 1

    for line in sys.stdin:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            command = json.loads(line)
            cmd = command.get("cmd")
            if cmd == "shutdown":
                emit({"event": "stopped", "message": "Worker stopped"})
                return 0
            if cmd == "infer":
                infer_one(
                    pose_model,
                    dataset,
                    dataset_info,
                    checkpoint_path,
                    device,
                    command,
                )
            else:
                emit({"event": "error", "message": f"Unknown command: {cmd}"})
        except Exception as exc:
            request_id = 0
            try:
                request_id = int(json.loads(line).get("request_id", 0))
            except Exception:
                pass
            emit(
                {
                    "event": "result",
                    "request_id": request_id,
                    "ok": False,
                    "message": f"Inference failed: {exc}",
                }
            )
            traceback.print_exc()

    emit({"event": "stopped", "message": "Worker input closed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
