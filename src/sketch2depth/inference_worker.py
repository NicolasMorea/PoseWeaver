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
    parser = argparse.ArgumentParser(description="Persistent Sketch2Depth inference worker")
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
    for key in ("ema_model", "model", "state_dict", "model_state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def load_model(checkpoint_path: Path, device_name: str):
    import torch

    from model_depth import DepthNet

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
    model = DepthNet(
        backbone_name=args.get(
            "backbone", args.get("backbone_name", "convnext_base.fb_in22k_ft_in1k")
        ),
        pretrained=False,
        decoder_channels=int(args.get("decoder_channels", 256)),
        dropout=float(args.get("dropout", 0.15)),
        drop_path_rate=float(args.get("drop_path_rate", 0.15)),
        max_detail_residual=float(args.get("max_detail_residual", 1.0)),
    ).to(device)

    state_dict = checkpoint_state_dict(ckpt)
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return (
        model,
        device,
        image_size,
        float(ckpt.get("depth_min", 0.0)) if isinstance(ckpt, dict) else 0.0,
        float(ckpt.get("depth_max", 1.0)) if isinstance(ckpt, dict) else 1.0,
    )


def preprocess_image(image_path: Path, image_size: int, device):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = np.transpose((array - mean) / std, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


def write_prediction(
    output_path: Path,
    sketch_path: Path,
    checkpoint_path: Path,
    pred: dict[str, Any],
    image_size: int,
    device,
    depth_min: float,
    depth_max: float,
) -> None:
    import numpy as np
    from PIL import Image

    depth = pred["depth"][0, 0].detach().cpu().numpy().astype(np.float32)
    mask = pred["mask"][0, 0].detach().cpu().numpy().astype(np.float32)
    boundary = pred["boundary"][0, 0].detach().cpu().numpy().astype(np.float32)

    stem = output_path.with_suffix("")
    render_path = stem.parent / (stem.name + "_render.png")
    depth_npz_path = stem.parent / (stem.name + "_depth.npz")
    render_path.parent.mkdir(parents=True, exist_ok=True)

    gray = np.clip(depth * 255.0, 0.0, 255.0).astype(np.uint8)
    alpha = ((mask > 0.5) * 255).astype(np.uint8)
    rgba = np.dstack((gray, gray, gray, alpha))
    Image.fromarray(rgba, mode="RGBA").save(render_path)
    np.savez_compressed(
        depth_npz_path,
        depth=depth,
        mask=mask,
        boundary=boundary,
        depth_min=np.asarray(depth_min, dtype=np.float32),
        depth_max=np.asarray(depth_max, dtype=np.float32),
        source_sketch=np.asarray(str(sketch_path.resolve())),
    )

    output = {
        "schema": "poseweaver.sketch2depth.v1",
        "source_sketch": str(sketch_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "image_size": image_size,
        "render_path": str(render_path.resolve()),
        "depth_npz_path": str(depth_npz_path.resolve()),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "channels": {
            "render": "RGBA PNG; RGB=normalized inverse depth, A=step(mask > 0.5)",
            "depth": "NPZ depth array shaped [H,W], normalized inverse depth (near=1, far=0)",
            "mask": "NPZ mask array shaped [H,W] with foreground probability in [0,1]",
            "boundary": "NPZ boundary array shaped [H,W] with boundary probability in [0,1]",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def infer_one(
    model,
    device,
    image_size: int,
    checkpoint_path: Path,
    depth_min: float,
    depth_max: float,
    command: dict[str, Any],
) -> None:
    import torch

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])
    emit(
        {
            "event": "busy",
            "request_id": request_id,
            "message": "Running Sketch2Depth inference",
        }
    )
    with torch.inference_mode():
        prediction = model(preprocess_image(sketch_path, image_size, device))
    write_prediction(
        output_path,
        sketch_path,
        checkpoint_path,
        prediction,
        image_size,
        device,
        depth_min,
        depth_max,
    )
    emit(
        {
            "event": "result",
            "request_id": request_id,
            "ok": True,
            "message": "Depth prediction complete",
            "output_path": str(output_path),
        }
    )


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    try:
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        model, device, image_size, depth_min, depth_max = load_model(
            checkpoint_path, args.device
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
            command_name = command.get("cmd")
            if command_name == "shutdown":
                emit({"event": "stopped", "message": "Worker stopped"})
                return 0
            if command_name == "infer":
                infer_one(
                    model,
                    device,
                    image_size,
                    checkpoint_path,
                    depth_min,
                    depth_max,
                    command,
                )
            else:
                emit({"event": "error", "message": f"Unknown command: {command_name}"})
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
