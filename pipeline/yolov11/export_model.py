"""
============================================================
YOLOv11 Model Export Script
============================================================
Export model YOLOv11 ke berbagai format untuk deployment.

Format yang didukung:
    onnx     - Open Neural Network Exchange (CPU/GPU universal)
    torchscript - TorchScript (PyTorch deployment)
    engine   - TensorRT (NVIDIA GPU, sangat cepat)
    openvino - OpenVINO (Intel CPU/GPU)
    coreml   - CoreML (Apple devices)
    tflite   - TensorFlow Lite (Mobile/Edge)

Cara Penggunaan:
    python export_model.py
    python export_model.py --format onnx
    python export_model.py --model path/to/best.pt --format tflite
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
    "pb": "TensorFlow GraphDef",
    "paddle": "PaddlePaddle",
    "ncnn": "NCNN (Mobile deployment)",
}


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Model Export")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--format", default="onnx", choices=list(FORMATS.keys()))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.model:
        model_path = args.model
    else:
        best = PROJECT_ROOT / "models" / "yolo v11" / "best.pt"
        if not best.exists():
            out = config.get("output", {})
            best = PROJECT_ROOT / out.get("project", "runs/yolov11") / out.get("name", "train") / "weights" / "best.pt"
        if not best.exists():
            print("ERROR: Model tidak ditemukan. Gunakan --model.")
            sys.exit(1)
        model_path = str(best)

    print(f"\n===== YOLOv11 Export =====\n")
    print(f"  Model   : {model_path}")
    print(f"  Format  : {args.format} - {FORMATS[args.format]}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  FP16    : {args.half}")
    print(f"  INT8    : {args.int8}")

    from ultralytics import YOLO
    model = YOLO(model_path)
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(f"\n  Export berhasil: {exported}")
    print("==========================\n")


if __name__ == "__main__":
    main()
