import os
from ultralytics import YOLO

# 1. Mendapatkan path ke model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "best.pt")

# 2. Load model
print("Loading model...")
model = YOLO(model_path)

# 3. Menampilkan informasi mengenai classes / label apa saja yang ada di dalam model
print("\n=== INFORMASI MODEL ===")
print("Daftar Class yang bisa dideteksi:", model.names)

# 4. Mengecek Matrik Akurasi (Membutuhkan Dataset)
# Untuk cek mAP, Precision, dan Recall, YOLO harus mengujinya pada dataset validation (.yaml)
# Anda perlu mengubah nilai argumen `data=` menggunakan path file konfigurasi `data.yaml` milik anda.
print("\n=== MENGHITUNG METRIC MODEL ===")
print("Menjalankan validasi evaluasi model...")

try:
    # Hilangkan tanda '#' pada 3 baris di bawah dan ubah file 'dataset.yaml' sesuai milik anda:
    
    # metrics = model.val(data='path/ke/dataset.yaml', split='val')
    
    # print(f"mAP 50       : {metrics.box.map50:.3f}")
    # print(f"mAP 50-95    : {metrics.box.map:.3f}")
    # print(f"Precision    : {metrics.box.mp:.3f}")
    # print(f"Recall       : {metrics.box.mr:.3f}")
    
    print("Silakan buka file `cek_metric.py` dan atur path ke file dataset `.yaml` Anda.")
except Exception as e:
    print(f"Tidak dapat menghitung metric: {e}")
    print("Pastikan Anda sudah mendefinisikan lokasi data.yaml Anda dengan benar.")
