"""
============================================================
YOLOv26 Validation & Metrics Script
============================================================
Mengevaluasi model YOLOv26 pada dataset validation atau test.

Catatan YOLOv26: Model ini bersifat NMS-Free (end-to-end).
Secara default menggunakan one-to-one head tanpa NMS.
Gunakan --end2end False untuk mode one-to-many (butuh NMS, akurasi sedikit lebih tinggi).

Cara Penggunaan:
    python validate_yolov26.py
    python validate_yolov26.py --model path/to/best.pt
    python validate_yolov26.py --split test
============================================================
"""
import sys, yaml, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Validation")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--end2end", type=str, default="True",
                        help="True=one-to-one (NMS-free), False=one-to-many (butuh NMS)")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Tentukan model path
    if args.model:
        model_path = args.model
    else:
        out = config.get("output", {})
        best_in_runs = PROJECT_ROOT / out.get("project", "runs/yolov26") / out.get("name", "train") / "weights" / "best.pt"
        best_in_models = PROJECT_ROOT / "models" / "yolo v26" / "best.pt"
        if best_in_models.exists():
            model_path = str(best_in_models)
        elif best_in_runs.exists():
            model_path = str(best_in_runs)
        else:
            print("ERROR: Model tidak ditemukan!")
            print(f"  Cek: {best_in_models}")
            print(f"  Atau: {best_in_runs}")
            sys.exit(1)

    ds_rel = config.get("dataset", "../../dataset.yaml")
    ds_path = str((SCRIPT_DIR / ds_rel).resolve())

    end2end = args.end2end.lower() in ("true", "1", "yes")

    print("\n===== YOLOv26 Validasi =====\n")
    print(f"  Model    : {model_path}")
    print(f"  Dataset  : {ds_path}")
    print(f"  Split    : {args.split}")
    print(f"  End2End  : {end2end} ({'NMS-free' if end2end else 'dengan NMS'})")
    print(f"  Conf     : {args.conf}")
    print(f"  IoU      : {args.iou}")

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
        project=str(PROJECT_ROOT / "runs" / "yolov26"),
        name=f"val_{args.split}",
        end2end=end2end,
    )

    print("\n===== HASIL METRIC =====\n")
    print(f"  mAP@50       : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95    : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")

    names = model.names
    print("\n  --- Per-Class ---")
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            cls_name = names.get(i, f"class_{i}")
            print(f"  {cls_name:15s}: AP50 = {ap:.4f}")

    print(f"\n  Hasil plot disimpan di: runs/yolov26/val_{args.split}/")
    print("===========================\n")


if __name__ == "__main__":
    main()
