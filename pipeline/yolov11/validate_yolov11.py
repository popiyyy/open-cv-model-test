"""
============================================================
YOLOv11 Validation & Metrics Script
============================================================
Mengevaluasi model YOLOv11 yang sudah di-training pada
dataset validation atau test.

Cara Penggunaan:
    python validate_yolov11.py
    python validate_yolov11.py --model path/to/best.pt
    python validate_yolov11.py --split test
============================================================
"""
import os, sys, yaml, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Validation")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    # Load config untuk mendapatkan default model dan dataset path
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Tentukan model path
    if args.model:
        model_path = args.model
    else:
        # Cari best.pt dari hasil training terakhir
        out = config.get("output", {})
        best_in_runs = PROJECT_ROOT / out.get("project", "runs/yolov11") / out.get("name", "train") / "weights" / "best.pt"
        best_in_models = PROJECT_ROOT / "models" / "yolo v11" / "best.pt"
        if best_in_models.exists():
            model_path = str(best_in_models)
        elif best_in_runs.exists():
            model_path = str(best_in_runs)
        else:
            print("ERROR: Model tidak ditemukan!")
            print(f"  Cek: {best_in_models}")
            print(f"  Atau: {best_in_runs}")
            print("  Gunakan --model untuk menentukan path manual.")
            sys.exit(1)

    # Resolve dataset path
    ds_rel = config.get("dataset", "../../dataset.yaml")
    ds_path = str((SCRIPT_DIR / ds_rel).resolve())

    print("\n===== YOLOv11 Validasi =====\n")
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {ds_path}")
    print(f"  Split   : {args.split}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")

    from ultralytics import YOLO
    model = YOLO(model_path)

    print("\nMenjalankan validasi...\n")
    metrics = model.val(
        data=ds_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "yolov11"),
        name=f"val_{args.split}",
    )

    print("\n===== HASIL METRIC =====\n")
    print(f"  mAP@50       : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95    : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")

    # Per-class metrics
    names = model.names
    print("\n  --- Per-Class ---")
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            cls_name = names.get(i, f"class_{i}")
            print(f"  {cls_name:15s}: AP50 = {ap:.4f}")

    print(f"\n  Hasil plot disimpan di: runs/yolov11/val_{args.split}/")
    print("===========================\n")


if __name__ == "__main__":
    main()
