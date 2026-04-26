"""
============================================================
YOLOv26 Training Pipeline - Script Utama
============================================================
YOLOv26 adalah evolusi terbaru YOLO dari Ultralytics:
- NMS-Free End-to-End inference (tanpa post-processing)
- MuSGD Optimizer (hybrid SGD + Muon)
- Hingga 43% lebih cepat di CPU
- DFL Removal untuk edge deployment

Cara Penggunaan:
    python train_yolov26.py
    python train_yolov26.py --epochs 50 --batch 8
    python train_yolov26.py --model yolo26m.pt
    python train_yolov26.py --resume path/to/last.pt
============================================================
"""
import os, sys, yaml, argparse, time, shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset(rel_path):
    p = (SCRIPT_DIR / rel_path).resolve()
    if not p.exists():
        print(f"ERROR: dataset tidak ditemukan: {p}")
        sys.exit(1)
    return str(p)


def validate_dataset(ds_path):
    print("\n--- Validasi Dataset ---")
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    for k in ["train", "val", "nc", "names"]:
        if k not in ds:
            print(f"ERROR: '{k}' tidak ada di dataset.yaml"); sys.exit(1)
    base = Path(ds_path).parent / ds.get("path", ".")
    for split in ["train", "val"]:
        sp = base / ds[split]
        n = len(list(sp.glob("*.*"))) if sp.exists() else 0
        status = "OK" if sp.exists() else "MISSING"
        print(f"  [{status}] {split}: {sp} ({n} files)")
        if not sp.exists(): sys.exit(1)
    if "test" in ds:
        tp = base / ds["test"]
        if tp.exists():
            print(f"  [OK] test: {tp} ({len(list(tp.glob('*.*')))} files)")
    nc = ds["nc"]
    names = list(ds["names"].values()) if isinstance(ds["names"], dict) else ds["names"]
    print(f"  Kelas ({nc}): {names}")
    print("  Validasi berhasil!\n")


def train(config, ds_path):
    from ultralytics import YOLO
    t = config.get("training", {})
    o = config.get("optimizer", {})
    a = config.get("augmentation", {})
    out = config.get("output", {})
    v = config.get("validation", {})
    device = t.get("device", "") or None
    out_project = str(PROJECT_ROOT / out.get("project", "runs/yolov26"))

    model_name = config.get("model", "yolo26n.pt")
    print(f"\nMemuat model: {model_name}")
    model = YOLO(model_name)
    start = time.time()

    results = model.train(
        data=ds_path,
        epochs=t.get("epochs", 100), batch=t.get("batch", 16),
        imgsz=t.get("imgsz", 640), patience=t.get("patience", 20),
        save_period=t.get("save_period", 10), workers=t.get("workers", 4),
        device=device,
        optimizer=o.get("name", "auto"), lr0=o.get("lr0", 0.01),
        lrf=o.get("lrf", 0.01), momentum=o.get("momentum", 0.937),
        weight_decay=o.get("weight_decay", 0.0005),
        warmup_epochs=o.get("warmup_epochs", 3.0),
        warmup_momentum=o.get("warmup_momentum", 0.8),
        hsv_h=a.get("hsv_h", 0.015), hsv_s=a.get("hsv_s", 0.7),
        hsv_v=a.get("hsv_v", 0.4), degrees=a.get("degrees", 0.0),
        translate=a.get("translate", 0.1), scale=a.get("scale", 0.5),
        shear=a.get("shear", 0.0), perspective=a.get("perspective", 0.0),
        flipud=a.get("flipud", 0.0), fliplr=a.get("fliplr", 0.5),
        mosaic=a.get("mosaic", 1.0), mixup=a.get("mixup", 0.0),
        copy_paste=a.get("copy_paste", 0.0), erasing=a.get("erasing", 0.4),
        close_mosaic=a.get("close_mosaic", 10),
        project=out_project, name=out.get("name", "train"),
        exist_ok=out.get("exist_ok", False), plots=out.get("plots", True),
        save=out.get("save", True), verbose=out.get("verbose", True),
        val=v.get("val", True), conf=v.get("conf", 0.001), iou=v.get("iou", 0.7),
    )
    elapsed = time.time() - start
    print(f"\nTraining selesai! Waktu: {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"Hasil: {out_project}/{out.get('name', 'train')}")
    return model, results


def main():
    print("\n===== YOLOv26 Training Pipeline =====\n")
    parser = argparse.ArgumentParser(description="YOLOv26 Training")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Path ke last.pt untuk resume")
    parser.add_argument("--name", default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    # Apply CLI overrides
    if args.model: config["model"] = args.model
    if args.epochs: config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch: config.setdefault("training", {})["batch"] = args.batch
    if args.imgsz: config.setdefault("training", {})["imgsz"] = args.imgsz
    if args.device: config.setdefault("training", {})["device"] = args.device
    if args.name: config.setdefault("output", {})["name"] = args.name

    ds_path = resolve_dataset(config.get("dataset", "../../dataset.yaml"))

    # Summary
    t = config.get("training", {})
    print(f"  Model     : {config.get('model', 'yolo26n.pt')}")
    print(f"  Dataset   : {ds_path}")
    print(f"  Epochs    : {t.get('epochs', 100)}")
    print(f"  Batch     : {t.get('batch', 16)}")
    print(f"  ImgSize   : {t.get('imgsz', 640)}")
    print(f"  Device    : {t.get('device', 'auto')}")

    if not args.skip_validation:
        validate_dataset(ds_path)

    # Resume
    if args.resume:
        from ultralytics import YOLO
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    resp = input("Mulai training? (y/n): ").strip().lower()
    if resp not in ("y", "yes", ""):
        print("Dibatalkan."); return

    model, results = train(config, ds_path)

    # Copy best.pt
    out = config.get("output", {})
    best = PROJECT_ROOT / out.get("project", "runs/yolov26") / out.get("name", "train") / "weights" / "best.pt"
    if best.exists():
        r = input("\nCopy best.pt ke models/yolo v26? (y/n): ").strip().lower()
        if r in ("y", "yes", ""):
            dst = PROJECT_ROOT / "models" / "yolo v26"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(best), str(dst / "best.pt"))
            print(f"Disalin ke: {dst / 'best.pt'}")

if __name__ == "__main__":
    main()
