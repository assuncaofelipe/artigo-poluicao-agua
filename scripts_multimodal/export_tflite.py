import os
import subprocess
import sys

import torch
from torch import nn

from config import Config
from model import SNNRegressor


def main():
    cfg = Config()
    ckpt = os.path.join(cfg.out_dir, "best_multi.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    base = SNNRegressor(
        input_size=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        output_size=len(cfg.target_cols),
        beta=cfg.beta,
        thresh=cfg.thresh,
    )
    base.load_state_dict(torch.load(ckpt, map_location="cpu"))
    base.eval()

    class ExportWrapper(nn.Module):
        def __init__(self, snn: nn.Module, time_steps: int):
            super().__init__()
            self.snn = snn
            self.time_steps = time_steps

        def forward(self, x):
            # deterministic rate encoding for export (no randomness)
            probs = torch.sigmoid(x)
            # use expand to avoid Tile op in ONNX
            spikes = probs.unsqueeze(0).expand(self.time_steps, -1, -1)
            return self.snn(spikes)

    model = ExportWrapper(base, cfg.time_steps)
    model.eval()

    dummy = torch.randn(1, len(cfg.feature_cols))
    onnx_path = "outputs/multimodal.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    # Convert ONNX -> TFLite using onnx2tf if available
    # Requires: pip install onnx onnx2tf tensorflow
    tflite_path = "outputs/model.tflite"
    def run_onnx2tf(extra_args=None):
        args = [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            onnx_path,
            "-o",
            "outputs/onnx2tf",
        ]
        if extra_args:
            args.extend(extra_args)
        subprocess.check_call(args)

    try:
        run_onnx2tf()
    except Exception:
        # if auto json exists, retry with -prf
        auto_json = "outputs/onnx2tf/multimodal_auto.json"
        if os.path.exists(auto_json):
            run_onnx2tf(["-prf", auto_json])
        else:
            raise

    # onnx2tf outputs a TFLite in the output folder
    candidates = [
        "outputs/onnx2tf/model_float32.tflite",
        "outputs/onnx2tf/multimodal_float32.tflite",
        "outputs/onnx2tf/model.tflite",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            os.replace(candidate, tflite_path)
            break
    print(f"TFLite saved to {tflite_path}")


if __name__ == "__main__":
    main()
