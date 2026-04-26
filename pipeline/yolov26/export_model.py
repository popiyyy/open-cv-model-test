"""
============================================================
YOLOv26 Model Export Script
============================================================
Export model YOLOv26 ke berbagai format untuk deployment.

Catatan YOLOv26: Secara default export menggunakan end-to-end mode
(NMS-free, one-to-one head). Gunakan --no-end2end untuk export
dengan one-to-many head (butuh NMS di sisi deployment).

Cara Penggunaan:
    python export_model.py
    python export_model.py --format onnx
    python export_model.py --format engine --half
    python export_model.py --format onnx --no-end2end
============================================================
"""
import sys, yaml, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FORMATS = {
    "onnx": "ONNX (Universal, direkomendasikan)",
    "torchscript": "TorchScript (PyTorch native)",
    "engine": "TensorRT (NVIDIA GPU, paling cepat)",
    "openvino": "OpenVINO (Intel hardware)",
    "coreml": "CoreML (Apple devices)",
    "tflite": "TFLite (Mobile / Edge devices)",
    "saved_model": "TensorFlow SavedModel",
    "ncnn": "NCNN (Mobile deployment)",
}


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Model Export")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--format", default="onnx", choices=list(FORMATS.keys()))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--no-end2end", action="store_true",
                        help="Export dengan one-to-many head (butuh NMS)")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.model:
        model_path = args.model
    else:
        best = PROJECT_ROOT / "models" / "yolo v26" / "best.pt"
        if not best.exists():
            out = config.get("output", {})
            best = PROJECT_ROOT / out.get("project", "runs/yolov26") / out.get("name", "train") / "weights" / "best.pt"
        if not best.exists():
            print("ERROR: Model tidak ditemukan. Gunakan --model.")
            sys.exit(1)
        model_path = str(best)

    end2end = not args.no_end2end

    print(f"\n===== YOLOv26 Export =====\n")
    print(f"  Model    : {model_path}")
    print(f"  Format   : {args.format} - {FORMATS[args.format]}")
    print(f"  ImgSize  : {args.imgsz}")
    print(f"  FP16     : {args.half}")
    print(f"  INT8     : {args.int8}")
    print(f"  End2End  : {end2end} ({'NMS-free' if end2end else 'dengan NMS'})")

    from ultralytics import YOLO
    model = YOLO(model_path)
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        end2end=end2end,
    )
    print(f"\n  Export berhasil: {exported}")
    print("==========================\n")


if __name__ == "__main__":
    main()
