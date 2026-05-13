#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


NUM_JOINTS = 22
JOINT_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1",
    "l_knee", "r_knee", "spine2",
    "l_ankle", "r_ankle", "neck",
    "l_foot", "r_foot", "head",
    "l_collar", "r_collar", "head_top",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def emit(payload: dict[str, Any]) -> None:
    print("POSEWEAVER_JSON " + json.dumps(payload, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Sketch2Rig inference worker")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--norm-stats", default="", help="Path to norm_stats.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def checkpoint_args(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        return ckpt["args"]
    return {}


def load_norm_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(checkpoint_path: Path, norm_stats_path: Path, device_name: str):
    import torch

    sys.path.insert(0, str(checkpoint_path.parent.resolve()))
    from model import Sketch3DPoseNet

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint_args(ckpt)

    model = Sketch3DPoseNet(
        backbone_name=args.get("backbone", "convnext_small.in12k_ft_in1k"),
        pretrained=False,
        d_model=int(args.get("d_model", 256)),
        nhead=int(args.get("nhead", 8)),
        num_decoder_layers=int(args.get("num_decoder_layers", 4)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)

    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    img_size = int(args.get("img_size", 256))
    norm_stats = ckpt.get("norm_stats", {}) if isinstance(ckpt, dict) else {}
    if not norm_stats:
        norm_stats = load_norm_stats(norm_stats_path)
    return model, device, img_size, norm_stats


def preprocess_image(image_path: Path, image_size: int, device):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # Matches the usual timm/ImageNet ConvNeXt preprocessing used for pretrained backbones.
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def write_prediction_json(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    prediction,
    image_size: int,
    device,
    norm_stats: dict[str, Any],
) -> None:
    joints = []
    raw = prediction.tolist()
    z_mean = norm_stats.get("z_mean")
    z_std = norm_stats.get("z_std")
    can_denormalize_z = isinstance(z_mean, (int, float)) and isinstance(z_std, (int, float))

    for i, xyz in enumerate(raw):
        position = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        if can_denormalize_z:
            position[2] = float(xyz[2]) * float(z_std) + float(z_mean)
        joint: dict[str, Any] = {
            "index": i,
            "name": JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}",
            "parent": SMPL_PARENTS[i] if i < len(SMPL_PARENTS) else -1,
            "position": position,
            "normalized_position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        }
        joints.append(joint)

    bones = [
        {
            "parent": parent,
            "child": child,
            "parent_name": JOINT_NAMES[parent],
            "child_name": JOINT_NAMES[child],
        }
        for child, parent in enumerate(SMPL_PARENTS)
        if parent >= 0
    ]

    output = {
        "schema": "poseweaver.sketch2rig.v1",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": image_size,
        "coordinate_space": "training_output_with_z_denormalized" if can_denormalize_z else "normalized_training_output",
        "norm_stats": norm_stats,
        "joints": joints,
        "bones": bones,
        "raw_prediction": raw,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp.replace(output_path)


def infer_one(model, device, image_size: int, norm_stats: dict[str, Any], checkpoint_path: Path, command: dict[str, Any]) -> None:
    import torch

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])

    emit({"event": "busy", "request_id": request_id, "message": "Running Sketch2Rig inference"})
    with torch.inference_mode():
        batch = preprocess_image(sketch_path, image_size, device)
        pred = model(batch)[0].detach().cpu()

    write_prediction_json(output_path, sketch_path, checkpoint_path, pred, image_size, device, norm_stats)
    emit({
        "event": "result",
        "request_id": request_id,
        "ok": True,
        "message": "Inference complete",
        "output_path": str(output_path),
    })


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    norm_stats_path = Path(args.norm_stats) if args.norm_stats else checkpoint_path.parent / "norm_stats.json"

    try:
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        model, device, image_size, norm_stats = load_model(checkpoint_path, norm_stats_path, args.device)
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
                infer_one(model, device, image_size, norm_stats, checkpoint_path, command)
            else:
                emit({"event": "error", "message": f"Unknown command: {cmd}"})
        except Exception as exc:
            request_id = 0
            try:
                request_id = int(json.loads(line).get("request_id", 0))
            except Exception:
                pass
            emit({
                "event": "result",
                "request_id": request_id,
                "ok": False,
                "message": f"Inference failed: {exc}",
            })
            traceback.print_exc()

    emit({"event": "stopped", "message": "Worker input closed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
