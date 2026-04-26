"""
============================================================
YOLOv26 Inference / Predict Script
============================================================
Menjalankan deteksi objek menggunakan model YOLOv26.

Catatan: YOLOv26 bersifat NMS-free secara default (end-to-end).
Prediksi dihasilkan langsung tanpa post-processing NMS.

Cara Penggunaan:
    python predict_yolov26.py --source 0                    # Webcam
    python predict_yolov26.py --source gambar.jpg           # Gambar
    python predict_yolov26.py --source folder_gambar/       # Folder
    python predict_yolov26.py --source video.mp4            # Video
============================================================
"""
import sys, yaml, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Inference")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--source", default="0", help="0=webcam, path gambar/video/folder")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold NMS")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Simpan hasil prediksi")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--device", default=None)
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

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"\n===== YOLOv26 Inference =====\n")
    print(f"  Model   : {model_path}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")

    from ultralytics import YOLO
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        device=args.device,
        stream=True,
    )

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"  Terdeteksi: {name} ({conf:.2f})")


if __name__ == "__main__":
    main()
