#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any


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
POSEWEAVER_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# NLF returns SMPL joints:
# pelvis, hips, spine, knees, ankles, feet, neck/collars/head, shoulders, elbows, wrists, hands.
NLF_TO_POSEWEAVER = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 12,
    10: 10,
    11: 11,
    12: 15,
    13: 13,
    14: 14,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
}


def emit(payload: dict[str, Any]) -> None:
    print("POSEWEAVER_JSON " + json.dumps(payload, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent NLF inference worker")
    parser.add_argument("--checkpoint", required=True, help="Path to nlf_l_multi.torchscript")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--body-model", choices=("smpl", "smplx"), default="smpl")
    parser.add_argument("--detector-threshold", type=float, default=0.30)
    parser.add_argument("--detector-nms-iou-threshold", type=float, default=0.70)
    parser.add_argument("--max-detections", type=int, default=150)
    parser.add_argument("--suppress-implausible-poses", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_rgb_tensor(path: Path, device):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device), image.size


def sketch_ink_bbox(path: Path) -> tuple[float, float, float, float] | None:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    luminance = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    ink = np.clip(1.0 - luminance, 0.0, 1.0)
    max_ink = float(np.max(ink)) if ink.size else 0.0
    if max_ink <= 1e-6:
        return None

    ys, xs = np.nonzero(ink > (max_ink / 3.0))
    if xs.size == 0 or ys.size == 0:
        return None
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def unwrap_single_image_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    unwrapped: dict[str, Any] = {}
    for key, value in prediction.items():
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Expected one batched result for {key}, got {len(value)}.")
            unwrapped[key] = value[0]
        else:
            unwrapped[key] = value
    return unwrapped


def tensor_to_numpy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return None


def select_person_index(prediction: dict[str, Any]) -> int | None:
    import numpy as np

    joints = tensor_to_numpy(prediction.get("joints2d"))
    if joints is None or joints.ndim != 3 or joints.shape[0] == 0:
        return None

    boxes = tensor_to_numpy(prediction.get("boxes"))
    if boxes is not None and boxes.ndim == 2 and boxes.shape[0] == joints.shape[0] and boxes.shape[1] >= 5:
        return int(np.nanargmax(boxes[:, 4]))
    return 0


def nlf_joint_arrays(prediction: dict[str, Any], person_index: int):
    import numpy as np

    joints2d = tensor_to_numpy(prediction.get("joints2d"))
    if joints2d is None or joints2d.ndim != 3 or person_index >= joints2d.shape[0]:
        return None, None

    joints3d = tensor_to_numpy(prediction.get("joints3d_nonparam"))
    if joints3d is None or joints3d.ndim != 3 or person_index >= joints3d.shape[0]:
        joints3d = tensor_to_numpy(prediction.get("joints3d"))

    xy = np.asarray(joints2d[person_index], dtype=np.float32)
    xyz = None
    if joints3d is not None and joints3d.ndim == 3 and person_index < joints3d.shape[0]:
        xyz = np.asarray(joints3d[person_index], dtype=np.float32)
    return xy, xyz


def nlf_betas(prediction: dict[str, Any], person_index: int) -> list[float]:
    betas = tensor_to_numpy(prediction.get("betas"))
    if betas is None or betas.ndim != 2 or person_index >= betas.shape[0]:
        return []
    return [safe_float(value) for value in betas[person_index].tolist()]


def nlf_pose_axis_angle(prediction: dict[str, Any], person_index: int) -> list[float]:
    import numpy as np

    pose = tensor_to_numpy(prediction.get("pose"))
    if pose is None or pose.ndim < 2 or person_index >= pose.shape[0]:
        return []

    values = np.asarray(pose[person_index], dtype=np.float32).reshape(-1)
    return [safe_float(value) for value in values.tolist()]


def nlf_flat_array(prediction: dict[str, Any], key: str, person_index: int) -> tuple[list[float], int]:
    import numpy as np

    values = tensor_to_numpy(prediction.get(key))
    if values is None or values.ndim != 3 or person_index >= values.shape[0] or values.shape[2] < 3:
        return [], 0

    person_values = np.asarray(values[person_index, :, :3], dtype=np.float32)
    return [safe_float(value) for value in person_values.reshape(-1).tolist()], int(person_values.shape[0])


def nlf_flat_xy_array(prediction: dict[str, Any], key: str, person_index: int) -> tuple[list[float], int]:
    import numpy as np

    values = tensor_to_numpy(prediction.get(key))
    if values is None or values.ndim != 3 or person_index >= values.shape[0] or values.shape[2] < 2:
        return [], 0

    person_values = np.asarray(values[person_index, :, :2], dtype=np.float32)
    return [safe_float(value) for value in person_values.reshape(-1).tolist()], int(person_values.shape[0])


def nlf_confidence(prediction: dict[str, Any], person_index: int | None) -> float:
    import numpy as np

    if person_index is None:
        return 0.0

    boxes = tensor_to_numpy(prediction.get("boxes"))
    if boxes is None or boxes.ndim != 2 or person_index >= boxes.shape[0] or boxes.shape[1] < 5:
        return 0.0

    return safe_float(float(np.asarray(boxes[person_index, 4]).item()))


def align_xy_to_sketch_bbox(xy, sketch_bbox: tuple[float, float, float, float] | None):
    import numpy as np

    if sketch_bbox is None or xy is None or xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return xy

    aligned = np.asarray(xy, dtype=np.float32).copy()
    finite = np.isfinite(aligned[:, 0]) & np.isfinite(aligned[:, 1])
    if not np.any(finite):
        return aligned

    pred_min_x = float(np.min(aligned[finite, 0]))
    pred_min_y = float(np.min(aligned[finite, 1]))
    pred_max_x = float(np.max(aligned[finite, 0]))
    pred_max_y = float(np.max(aligned[finite, 1]))
    pred_w = max(1e-6, pred_max_x - pred_min_x)
    pred_h = max(1e-6, pred_max_y - pred_min_y)

    sketch_min_x, sketch_min_y, sketch_max_x, sketch_max_y = sketch_bbox
    sketch_w = max(1e-6, sketch_max_x - sketch_min_x)
    sketch_h = max(1e-6, sketch_max_y - sketch_min_y)

    scale_x = max(0.05, min(20.0, sketch_w / pred_w))
    scale_y = max(0.05, min(20.0, sketch_h / pred_h))
    pred_cx = (pred_min_x + pred_max_x) * 0.5
    pred_cy = (pred_min_y + pred_max_y) * 0.5
    sketch_cx = (sketch_min_x + sketch_max_x) * 0.5
    sketch_cy = (sketch_min_y + sketch_max_y) * 0.5

    aligned[:, 0] = (aligned[:, 0] - pred_cx) * scale_x + sketch_cx
    aligned[:, 1] = (aligned[:, 1] - pred_cy) * scale_y + sketch_cy
    return aligned


def safe_float(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else fallback


def mapped_depths(xyz, joint_count: int) -> list[float]:
    # The app uses these joints as a sketch-alignment rig. NLF's predicted 3D
    # joints are still represented by pose_axis_angle; feeding raw NLF depth
    # into this rig can create an oversized perspective skeleton.
    return [2.5] * joint_count


def build_poseweaver_joints(xy, xyz, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    import numpy as np

    if xy is None or xy.ndim != 2 or xy.shape[0] < 22 or xy.shape[1] < 2:
        return []

    image_width, image_height = image_size
    inv_w = 1.0 / max(1.0, float(image_width - 1))
    inv_h = 1.0 / max(1.0, float(image_height - 1))
    depths = mapped_depths(xyz, xy.shape[0])
    mapped: list[list[float] | None] = [None] * len(JOINT_NAMES)

    for pw_index, nlf_index in NLF_TO_POSEWEAVER.items():
        if nlf_index >= xy.shape[0]:
            continue
        x = safe_float(xy[nlf_index, 0]) * inv_w
        y = safe_float(xy[nlf_index, 1]) * inv_h
        mapped[pw_index] = [x, y, depths[nlf_index]]

    if mapped[9] is None and xy.shape[0] > 12:
        mapped[9] = [safe_float(xy[12, 0]) * inv_w, safe_float(xy[12, 1]) * inv_h, depths[12]]
    if mapped[12] is None and xy.shape[0] > 15:
        mapped[12] = [safe_float(xy[15, 0]) * inv_w, safe_float(xy[15, 1]) * inv_h, depths[15]]
    if mapped[15] is None and mapped[9] is not None and mapped[12] is not None:
        neck = np.asarray(mapped[9], dtype=np.float32)
        head = np.asarray(mapped[12], dtype=np.float32)
        mapped[15] = (head + 0.6 * (head - neck)).astype(np.float32).tolist()
    elif mapped[15] is None and mapped[12] is not None:
        mapped[15] = list(mapped[12])

    joints: list[dict[str, Any]] = []
    for index, position in enumerate(mapped):
        if position is None:
            return []
        joints.append(
            {
                "index": index,
                "name": JOINT_NAMES[index],
                "parent": POSEWEAVER_PARENTS[index],
                "position": [safe_float(position[0]), safe_float(position[1]), safe_float(position[2], 2.5)],
            }
        )
    return joints


def write_prediction_json(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    device,
    image_size: tuple[int, int],
    joints: list[dict[str, Any]],
    betas: list[float],
    pose_axis_angle: list[float],
    vertices3d_flat: list[float],
    vertices3d_count: int,
    vertices2d_flat: list[float],
    vertices2d_count: int,
    joints3d_flat: list[float],
    joints3d_count: int,
    confidence: float,
    person_count: int,
) -> None:
    bones = [
        {
            "parent": parent,
            "child": child,
            "parent_name": JOINT_NAMES[parent],
            "child_name": JOINT_NAMES[child],
        }
        for child, parent in enumerate(POSEWEAVER_PARENTS)
        if parent >= 0
    ]

    image_width, image_height = image_size
    result = {
        "schema": "poseweaver.nlf.v1",
        "source_model": "NLF",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": max(int(image_width), int(image_height)),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "coordinate_space": "normalized_nlf_projected_2d_flat_depth_with_nlf_pose_axis_angle",
        "person_count": int(person_count),
        "confidence": safe_float(confidence),
        "joints": joints,
        "bones": bones,
        "betas": betas,
        "pose_axis_angle": pose_axis_angle,
        "vertices3d_flat": vertices3d_flat,
        "vertices3d_count": int(vertices3d_count),
        "vertices2d_flat": vertices2d_flat,
        "vertices2d_count": int(vertices2d_count),
        "joints3d_flat": joints3d_flat,
        "joints3d_count": int(joints3d_count),
        "gender": "neutral",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def infer_one(model, checkpoint_path: Path, device, args: argparse.Namespace, command: dict[str, Any]) -> None:
    import torch

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])

    emit({"event": "busy", "request_id": request_id, "message": "Running NLF inference"})
    with torch.inference_mode(), torch.device(device):
        frame, image_size = load_rgb_tensor(sketch_path, device)
        prediction = unwrap_single_image_prediction(
            model.detect_smpl_batched(
                frame.unsqueeze(0),
                detector_threshold=args.detector_threshold,
                detector_nms_iou_threshold=args.detector_nms_iou_threshold,
                max_detections=args.max_detections,
                suppress_implausible_poses=args.suppress_implausible_poses,
                model_name=args.body_model,
            )
        )

    person_count = 0
    xy = None
    xyz = None
    betas: list[float] = []
    pose_axis_angle: list[float] = []
    vertices3d_flat: list[float] = []
    vertices3d_count = 0
    vertices2d_flat: list[float] = []
    vertices2d_count = 0
    joints3d_flat: list[float] = []
    joints3d_count = 0
    person_index = select_person_index(prediction)
    confidence = nlf_confidence(prediction, person_index)
    if person_index is not None:
        person_count = int(tensor_to_numpy(prediction.get("joints2d")).shape[0])
        xy, xyz = nlf_joint_arrays(prediction, person_index)
        betas = nlf_betas(prediction, person_index)
        pose_axis_angle = nlf_pose_axis_angle(prediction, person_index)
        vertices3d_flat, vertices3d_count = nlf_flat_array(prediction, "vertices3d", person_index)
        vertices2d_flat, vertices2d_count = nlf_flat_xy_array(prediction, "vertices2d", person_index)
        joints3d_flat, joints3d_count = nlf_flat_array(prediction, "joints3d", person_index)

    joints = build_poseweaver_joints(xy, xyz, image_size)
    write_prediction_json(
        output_path,
        sketch_path,
        checkpoint_path,
        device,
        image_size,
        joints,
        betas,
        pose_axis_angle,
        vertices3d_flat,
        vertices3d_count,
        vertices2d_flat,
        vertices2d_count,
        joints3d_flat,
        joints3d_count,
        confidence,
        person_count)
    emit(
        {
            "event": "result",
            "request_id": request_id,
            "ok": True,
            "message": f"NLF bones predicted ({len(joints)} joints, confidence {confidence:.3f})",
            "output_path": str(output_path),
        }
    )


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    try:
        import torch
        import torchvision  # Required before torch.jit.load for torchvision custom ops.

        device = resolve_device(args.device)
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        model = torch.jit.load(str(checkpoint_path)).to(device).eval()
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
                infer_one(model, checkpoint_path, device, args, command)
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
