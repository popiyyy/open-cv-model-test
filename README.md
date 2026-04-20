# YOLOv8 OpenCV Inference Project

Proyek ini berisi kumpulan script Python untuk memuat (*load*), menguji metrik, menjalankan inferensi secara *real-time* dengan webcam, serta membandingkan performa model deteksi objek (YOLOv8) menggunakan pendekatan berbasis OpenCV.

---

## 📁 Struktur Direktori

```text
.
├── models/             # Folder tempat meletakkan file model (.pt), contoh: best.pt, model2.pt
├── scripts/
│   ├── run_model.py    # Script mendeteksi objek secara live via webcam menggunakan 1 model
│   ├── compare_model.py# Script untuk membandingkan 2 model secara visual (kiri-kanan)
│   └── cek_metric.py   # Script untuk mengecek angka evaluasi/metric (mAP, Precision, Recall)
├── requirements.txt    # Daftar dependensi module Python yang dibutuhkan
└── .gitignore          # File untuk mengecualikan environment dan file beban besar dari Git
```

---

## 🛠️ Persiapan & Instalasi

1. **Membuat dan Mengaktifkan Virtual Environment**  
   Disarankan untuk menjalankan kode dalam *virtual environment* agar rapi:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Untuk Windows
   ```

2. **Menginstal Dependensi**  
   Jalankan file *requirements* untuk menginstal YOLO (Ultralytics), OpenCV, dan sebagainya:
   ```bash
   pip install -r requirements.txt
   ```

3. **Menyiapkan Model YOLO**  
   Letakkan file hasil *training* YOLO Anda (biasanya ber-ekstensi `.pt`) ke dalam folder `models/`.

---

## 🚀 Penggunaan Script

### 1. Deteksi *Real-time* (Webcam Tunggal)
Menggunakan webcam komputer Anda untuk mendeteksi objek.
```bash
python scripts/run_model.py
```
> **Tip:** Anda dapat menghentikan sesi kamera kapan saja dengan menekan tombol `q` pada *keyboard*. Jika ingin mengubah batas keyakinan deteksi (*confidence*), edit nilai `conf=0.5` di dalam script.

### 2. Membandingkan 2 Model Sekaligus (*Side-by-Side*)
Jika Anda memiliki beberapa iterasi / variasi model (misal `best.pt` dan `model2.pt`) dan bimbang model mana yang berfungsi lebih baik:
```bash
python scripts/compare_model.py
```
Script ini akan membuka webcam dan memproses frame yang sama untuk dilemparkan kepada kedua model, lalu menggabungkannya dalam satu layar secara bersebelahan (*split mode*).

### 3. Mengevaluasi Metrik Akurasi
Ingin melihat detail akurasi model seperti mAP50, Precision, dan Recall?
1. Edit file `scripts/cek_metric.py`. Hapus komentar pada *try-catch statement*.
2. Arahkan *path* dataset yang ditandai dengan tulisan `data='path/ke/dataset.yaml'`. *(Dibutuhkan file data.yaml sisa *training* model yang digunakan).*
3. Jalankan:
```bash
python scripts/cek_metric.py
```

---

## 📝 Catatan Penting
- Di dalam konfigurasi `.gitignore`, folder `models/` beserta isinya akan diabaikan oleh Git. Hal ini dilakukan karena beban timbangan file AI (`.pt`) relatif raksasa dan melebihi batasan ukuran upload dari *repository* seperti GitHub. Jika ingin tetap memasukkan model, Anda dapat menyesuaikan baris abaian di `.gitignore`.
