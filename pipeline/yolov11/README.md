# 🚀 YOLOv11 Training Pipeline

Pipeline lengkap untuk **training**, **validasi**, **export**, dan **inference** model YOLOv11 untuk deteksi objek.

---

## 📁 Struktur File

```
pipeline/yolov11/
├── config.yaml                                   # Konfigurasi hyperparameter training
├── train_yolov11.py                              # Script utama training
├── validate_yolov11.py                           # Evaluasi metric model
├── export_model.py                               # Export model ke format deployment
├── predict_yolov11.py                            # Inference / deteksi objek
├── yolov11_roboflow_training_template.ipynb      # Template training (Google Colab)
└── README.md                                     # Dokumentasi (file ini)
```

---

## ⚡ Quick Start

### 1. Pastikan dependencies terinstall
```bash
pip install ultralytics>=8.2.0 pyyaml
```

### 2. Jalankan training
```bash
cd pipeline/yolov11
python train_yolov11.py
```

### 3. Evaluasi hasil
```bash
python validate_yolov11.py
```

### 4. Jalankan prediksi
```bash
python predict_yolov11.py --source 0    # Webcam
```

---

## 🏋️ Training

### Penggunaan Dasar
```bash
python train_yolov11.py
```

### Override Parameter via CLI
```bash
# Ubah epochs dan batch size
python train_yolov11.py --epochs 50 --batch 8

# Gunakan model yang lebih besar
python train_yolov11.py --model yolo11m.pt

# Training pakai CPU
python train_yolov11.py --device cpu

# Resume training dari checkpoint
python train_yolov11.py --resume runs/yolov11/train/weights/last.pt
```

### Pilihan Model YOLOv11
| Model         | Ukuran  | Kecepatan | Akurasi |
|:------------- |:-------:|:---------:|:-------:|
| `yolo11n.pt`  | Nano    | ⚡⚡⚡⚡⚡   | ⭐⭐      |
| `yolo11s.pt`  | Small   | ⚡⚡⚡⚡    | ⭐⭐⭐     |
| `yolo11m.pt`  | Medium  | ⚡⚡⚡     | ⭐⭐⭐⭐    |
| `yolo11l.pt`  | Large   | ⚡⚡      | ⭐⭐⭐⭐⭐   |
| `yolo11x.pt`  | XLarge  | ⚡       | ⭐⭐⭐⭐⭐⭐  |

---

## 🧪 Validasi

```bash
# Validasi pada val split (default)
python validate_yolov11.py

# Validasi pada test split
python validate_yolov11.py --split test

# Gunakan model tertentu
python validate_yolov11.py --model path/to/best.pt
```

**Output metric:** mAP@50, mAP@50-95, Precision, Recall, dan per-class AP.

---

## 📦 Export Model

```bash
# Export ke ONNX (default, direkomendasikan)
python export_model.py

# Export ke TensorRT (NVIDIA GPU)
python export_model.py --format engine

# Export ke TFLite (mobile)
python export_model.py --format tflite

# Export dengan FP16 quantization
python export_model.py --format onnx --half
```

---

## 🎯 Inference / Prediksi

```bash
# Webcam real-time
python predict_yolov11.py --source 0

# Satu gambar
python predict_yolov11.py --source path/to/image.jpg

# Folder gambar
python predict_yolov11.py --source path/to/folder/

# Video
python predict_yolov11.py --source path/to/video.mp4

# Atur confidence threshold
python predict_yolov11.py --source 0 --conf 0.7
```

---

## ⚙️ Konfigurasi (config.yaml)

Semua hyperparameter training dikonfigurasi di `config.yaml`:

- **Model**: Pilihan varian model (nano sampai xlarge)
- **Training**: epochs, batch size, image size, early stopping
- **Optimizer**: learning rate, momentum, weight decay, warmup
- **Augmentation**: mosaic, flip, HSV, scale, rotate, mixup
- **Output**: folder output, nama eksperimen
- **Validation**: split, confidence, IoU threshold

---

## 🔄 Alur Kerja (Workflow)

```
1. Siapkan Dataset (images/ + labels/ dalam format YOLO)
         ↓
2. Konfigurasi dataset.yaml (path, kelas, nama)
         ↓
3. Sesuaikan config.yaml (hyperparameter)
         ↓
4. Jalankan train_yolov11.py (training)
         ↓
5. Jalankan validate_yolov11.py (evaluasi)
         ↓
6. Copy best.pt → models/yolo v11/
         ↓
7. Jalankan predict_yolov11.py (test real-time)
         ↓
8. (Opsional) export_model.py untuk deployment
```

---

## 💡 Tips

- **GPU Memory Error?** Turunkan `batch` di config.yaml (coba 8 atau 4)
- **Training lambat?** Gunakan model lebih kecil (`yolo11n.pt`)
- **Akurasi rendah?** Tambah epochs, gunakan model lebih besar, atau tambah data augmentation
- **Overfitting?** Turunkan `patience`, tambah augmentation, atau gunakan dataset lebih banyak
