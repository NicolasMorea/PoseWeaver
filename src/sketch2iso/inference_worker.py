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
    parser = argparse.ArgumentParser(description="Persistent Sketch2ISO inference worker")
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
    import numpy as np
    import torch

    from model_iso import ISONet

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

    model = ISONet(
        backbone_name=args.get("backbone", args.get("backbone_name", "convnext_base.fb_in22k_ft_in1k")),
        pretrained=False,
        decoder_channels=int(args.get("decoder_channels", 256)),
        dropout=float(args.get("dropout", 0.15)),
        drop_path_rate=float(args.get("drop_path_rate", 0.15)),
    ).to(device)

    state_dict = checkpoint_state_dict(ckpt)
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    descriptor_min = np.asarray(ckpt.get("descriptor_min"), dtype=np.float32)
    descriptor_max = np.asarray(ckpt.get("descriptor_max"), dtype=np.float32)
    if descriptor_min.shape != (64,) or descriptor_max.shape != (64,):
        raise ValueError("Checkpoint is missing 64D descriptor_min/descriptor_max")

    return model, device, image_size, descriptor_min, descriptor_max


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


def write_prediction(
    output_path: Path,
    sketch_path: Path,
    pred: dict[str, Any],
    descriptor_min,
    descriptor_max,
) -> None:
    import numpy as np

    iso_normalized = pred["iso"][0].detach().cpu().numpy()
    mask_probability = pred["mask"][0, 0].detach().cpu().numpy()
    descriptor_range = descriptor_max - descriptor_min
    isomap = iso_normalized * descriptor_range[:, None, None] + descriptor_min[:, None, None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            isomap=np.transpose(isomap, (1, 2, 0)).astype(np.float32),
            mask=(mask_probability > 0.5),
            mask_probability=mask_probability.astype(np.float32),
            dimensions=np.asarray(64, dtype=np.int64),
            layout=np.asarray("HWC"),
            source_sketch=np.asarray(str(sketch_path.resolve())),
            descriptor_min=descriptor_min.astype(np.float32),
            descriptor_max=descriptor_max.astype(np.float32),
        )
    tmp_path.replace(output_path)


def infer_one(
    model,
    device,
    image_size: int,
    descriptor_min,
    descriptor_max,
    command: dict[str, Any],
) -> None:
    import torch

    request_id = int(command.get("request_id", 0))
    sketch_path = Path(command["sketch_path"])
    output_path = Path(command["output_path"])

    emit({"event": "busy", "request_id": request_id, "message": "Running Sketch2ISO inference"})
    with torch.inference_mode():
        batch = preprocess_image(sketch_path, image_size, device)
        pred = model(batch)

    write_prediction(output_path, sketch_path, pred, descriptor_min, descriptor_max)
    emit({
        "event": "result",
        "request_id": request_id,
        "ok": True,
        "message": "ISO prediction complete",
        "output_path": str(output_path),
    })


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    try:
        emit({"event": "loading", "message": f"Loading model: {checkpoint_path}"})
        model, device, image_size, descriptor_min, descriptor_max = load_model(
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
            cmd = command.get("cmd")
            if cmd == "shutdown":
                emit({"event": "stopped", "message": "Worker stopped"})
                return 0
            if cmd == "infer":
                infer_one(
                    model,
                    device,
                    image_size,
                    descriptor_min,
                    descriptor_max,
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
