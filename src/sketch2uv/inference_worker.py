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
    parser = argparse.ArgumentParser(description="Persistent Sketch2UV inference worker")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def checkpoint_args(ckpt: Any) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        return {}
    for key in ("args", "config", "hparams"):
        args = ckpt.get(key, {})
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


def load_model(checkpoint_path: Path, device_name: str):
    import torch

    from model_uvpred import UVPredNet

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint_args(ckpt)
    image_size = int(args.get("img_size", args.get("image_size", 512)))

    model = UVPredNet(
        backbone_name=args.get("backbone", args.get("backbone_name", "convnext_base.fb_in22k_ft_in1k")),
        pretrained=False,
        decoder_channels=int(args.get("decoder_channels", 256)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)

    state_dict = checkpoint_state_dict(ckpt)
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device, image_size


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


def write_png(path: Path, rgba) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def write_prediction_files(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    pred: dict[str, Any],
    image_size: int,
    device,
) -> None:
    import numpy as np

    alpha_threshold = 0.5

    uv = pred["uv"][0].detach().cpu().numpy()
    mask = pred["mask"][0, 0].detach().cpu().numpy()
    render = pred["render"][0].detach().cpu().numpy()

    mask_step = (mask > alpha_threshold).astype(np.float32)
    alpha = (mask_step * 255.0).astype(np.uint8)
    uv_rgb = np.concatenate([uv, np.zeros_like(mask[None, :, :])], axis=0)
    render_rgb_f = np.transpose(uv_rgb, (1, 2, 0)) * mask_step[:, :, None]
    render_rgb = np.clip(render_rgb_f * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack([render_rgb, alpha])

    stem = output_path.with_suffix("")
    render_path = stem.parent / (stem.name + "_render.png")
    uv_npz_path = stem.parent / (stem.name + "_uv.npz")

    write_png(render_path, rgba)
    uv_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        uv_npz_path,
        uv=uv.astype(np.float32),
        mask=mask.astype(np.float32),
        render=render.astype(np.float32),
    )

    output = {
        "schema": "poseweaver.sketch2uv.v1",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": image_size,
        "render_path": str(render_path.resolve()),
        "uv_npz_path": str(uv_npz_path.resolve()),
        "channels": {
            "render": "RGBA PNG; R=U, G=V, B=0, A=step(mask > 0.85)",
            "uv": "NPZ uv array shaped [2,H,W] with normalized U/V in [0,1]",
            "mask": "NPZ mask array shaped [H,W] with foreground probability in [0,1]",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tmp.replace(output_path)


def infer_one(model, device, image_size: int, checkpoint_path: Path, command: dict[str, Any]) -> None:
    import torch

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])

    emit({"event": "busy", "request_id": request_id, "message": "Running Sketch2UV inference"})
    with torch.inference_mode():
        batch = preprocess_image(sketch_path, image_size, device)
        pred = model(batch)

    write_prediction_files(output_path, sketch_path, checkpoint_path, pred, image_size, device)
    emit({
        "event": "result",
        "request_id": request_id,
        "ok": True,
        "message": "UV prediction complete",
        "output_path": str(output_path),
    })


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    try:
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        model, device, image_size = load_model(checkpoint_path, args.device)
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
                infer_one(model, device, image_size, checkpoint_path, command)
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
