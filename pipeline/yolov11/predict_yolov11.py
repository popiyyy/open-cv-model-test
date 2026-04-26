"""
============================================================
YOLOv11 Inference / Predict Script
============================================================
Menjalankan prediksi/deteksi objek menggunakan model YOLOv11
pada gambar, video, atau webcam.

Cara Penggunaan:
    python predict_yolov11.py --source 0                    # Webcam
    python predict_yolov11.py --source gambar.jpg           # Gambar
    python predict_yolov11.py --source folder_gambar/       # Folder
    python predict_yolov11.py --source video.mp4            # Video
    python predict_yolov11.py --source 0 --conf 0.5         # Webcam + confidence
============================================================
"""
import sys, yaml, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Inference")
    parser.add_argument("--model", default=None, help="Path ke model .pt")
    parser.add_argument("--source", default="0", help="Sumber: 0 (webcam), path gambar/video/folder")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold NMS")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", help="Simpan hasil prediksi")
    parser.add_argument("--show", action="store_true", default=True, help="Tampilkan hasil di layar")
    parser.add_argument("--device", default=None)
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

    # Convert source "0" ke integer untuk webcam
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"\n===== YOLOv11 Inference =====\n")
    print(f"  Model   : {model_path}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")

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

    # Process results (needed for stream=True)
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
