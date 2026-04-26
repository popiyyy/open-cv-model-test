# 🚀 YOLOv26 Training Pipeline

Pipeline lengkap untuk **training**, **validasi**, **export**, dan **inference** model YOLOv26 untuk deteksi objek.

| Fitur | Deskripsi |
|:------|:----------|
| **NMS-Free** | End-to-end inference, tanpa post-processing NMS |
| **MuSGD Optimizer** | Hybrid SGD + Muon (dari Kimi K2) untuk training lebih stabil |
| **43% Lebih Cepat** | Inference CPU jauh lebih cepat dari YOLO11 |
| **DFL Removal** | Arsitektur lebih sederhana, cocok untuk edge device |
| **Dual-Head** | One-to-one (NMS-free) atau one-to-many (akurasi lebih tinggi) |

---

## 📁 Struktur File

```
pipeline/yolov26/
├── config.yaml                              # Konfigurasi hyperparameter
├── train_yolov26.py                         # Script training (lokal)
├── validate_yolov26.py                      # Evaluasi metric
├── export_model.py                          # Export model ke ONNX/TensorRT/dll
├── predict_yolov26.py                       # Inference / deteksi objek
├── yolov26_roboflow_training_template.ipynb  # Notebook Google Colab
└── README.md                                # Dokumentasi (file ini)
```

---

## 🏋️ Quick Start (Lokal)

```bash
cd pipeline/yolov26

# Training
python train_yolov26.py

# Override parameter
python train_yolov26.py --model yolo26s.pt --epochs 50 --batch 8

# Validasi
python validate_yolov26.py --split test

# Prediksi webcam
python predict_yolov26.py --source 0

# Export ke ONNX
python export_model.py --format onnx
```

## ☁️ Quick Start (Google Colab)

1. Upload `yolov26_roboflow_training_template.ipynb` ke Google Colab
2. Ubah Runtime ke **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Jalankan cell satu per satu
4. Hasil otomatis tersimpan ke Google Drive

---

## 📊 Pilihan Model YOLOv26

| Model | Ukuran | Kecepatan | Akurasi |
|:------|:------:|:---------:|:-------:|
| `yolo26n.pt` | Nano | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| `yolo26s.pt` | Small | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `yolo26m.pt` | Medium | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `yolo26l.pt` | Large | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| `yolo26x.pt` | XLarge | ⚡ | ⭐⭐⭐⭐⭐⭐ |

---

## 🔄 Dual-Head Architecture (Fitur Baru)

YOLOv26 memiliki 2 mode prediksi:

- **One-to-One (default):** NMS-free, lebih cepat, output maks 300 deteksi
- **One-to-Many:** Butuh NMS, akurasi sedikit lebih tinggi

```bash
# Validasi dengan one-to-many head
python validate_yolov26.py --end2end False

# Export dengan one-to-many head
python export_model.py --no-end2end
```

---

## 💡 Tips

- **GPU Memory Error?** Turunkan `batch` (coba 8 atau 4)
- **Training lambat?** Gunakan model lebih kecil (`yolo26n.pt`)
- **Akurasi rendah?** Tambah epochs, gunakan model lebih besar, atau tambah data
- **Edge deployment?** Gunakan `yolo26n.pt` + export ke ONNX/TFLite
