#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    print("POSEWEAVER_JSON " + json.dumps(payload, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Sketch2Betas inference worker")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--norm-stats", default="", help="Path to betas_norm_stats.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def checkpoint_args(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        args = ckpt.get("args", {})
        if isinstance(args, dict):
            return args
        if hasattr(args, "__dict__"):
            return vars(args)
    return {}


def checkpoint_state_dict(ckpt: Any) -> Any:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model", "state_dict", "model_state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def load_norm_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(checkpoint_path: Path, norm_stats_path: Path, device_name: str):
    import torch

    from model_betas import NUM_BETAS, SketchBetasGenderNet

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint_args(ckpt)
    norm_stats = ckpt.get("norm_stats", {}) if isinstance(ckpt, dict) else {}
    if not norm_stats:
        norm_stats = load_norm_stats(norm_stats_path)

    beta_mean = norm_stats.get("beta_mean")
    gender_to_idx = norm_stats.get("gender_to_idx", {})
    num_betas = int(args.get("num_betas", len(beta_mean) if isinstance(beta_mean, list) else NUM_BETAS))
    num_genders = int(args.get("num_genders", len(gender_to_idx) if isinstance(gender_to_idx, dict) else 2))
    image_size = int(args.get("img_size", args.get("image_size", 384)))

    model = SketchBetasGenderNet(
        backbone_name=args.get("backbone", args.get("backbone_name", "convnext_base.fb_in22k_ft_in1k")),
        num_betas=num_betas,
        num_genders=num_genders,
        pretrained=False,
        d_model=int(args.get("d_model", 512)),
        nhead=int(args.get("nhead", 8)),
        dropout=float(args.get("dropout", 0.15)),
        mlp_blocks=int(args.get("mlp_blocks", 2)),
        img_size=image_size,
    ).to(device)

    state_dict = checkpoint_state_dict(ckpt)
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device, image_size, norm_stats


def preprocess_image(image_path: Path, image_size: int, device):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def denormalize_betas(normalized: list[float], norm_stats: dict[str, Any]) -> list[float]:
    mean = norm_stats.get("beta_mean")
    std = norm_stats.get("beta_std")
    if not isinstance(mean, list) or not isinstance(std, list):
        return [float(v) for v in normalized]

    betas: list[float] = []
    for i, value in enumerate(normalized):
        m = float(mean[i]) if i < len(mean) else 0.0
        s = float(std[i]) if i < len(std) else 1.0
        betas.append(float(value) * s + m)
    return betas


def idx_to_gender(norm_stats: dict[str, Any]) -> dict[int, str]:
    mapping = norm_stats.get("gender_to_idx", {})
    if isinstance(mapping, dict):
        out: dict[int, str] = {}
        for name, idx in mapping.items():
            try:
                out[int(idx)] = str(name)
            except Exception:
                pass
        if out:
            return out
    return {0: "female", 1: "male"}


def write_prediction_json(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    normalized_betas: list[float],
    gender_logits: list[float],
    image_size: int,
    device,
    norm_stats: dict[str, Any],
) -> None:
    import torch

    probs = torch.softmax(torch.tensor(gender_logits), dim=0).tolist() if gender_logits else []
    gender_index = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else -1
    gender = idx_to_gender(norm_stats).get(gender_index, "unknown")
    betas = denormalize_betas(normalized_betas, norm_stats)

    output = {
        "schema": "poseweaver.sketch2betas.v1",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": image_size,
        "betas": betas,
        "normalized_betas": normalized_betas,
        "gender": gender,
        "gender_index": gender_index,
        "gender_probabilities": probs,
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

    emit({"event": "busy", "request_id": request_id, "message": "Running Sketch2Betas inference"})
    with torch.inference_mode():
        batch = preprocess_image(sketch_path, image_size, device)
        pred = model(batch)
        normalized_betas = pred["betas"][0].detach().cpu().tolist()
        gender_logits = pred["gender_logits"][0].detach().cpu().tolist()

    write_prediction_json(
        output_path,
        sketch_path,
        checkpoint_path,
        [float(v) for v in normalized_betas],
        [float(v) for v in gender_logits],
        image_size,
        device,
        norm_stats,
    )
    emit({
        "event": "result",
        "request_id": request_id,
        "ok": True,
        "message": "Betas extracted",
        "output_path": str(output_path),
    })


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    norm_stats_path = Path(args.norm_stats) if args.norm_stats else checkpoint_path.parent / "betas_norm_stats.json"

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
