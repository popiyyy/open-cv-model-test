import cv2
import os
import numpy as np
from ultralytics import YOLO

# 1. Tentukan path ke kedua model yang ingin dibandingkan
script_dir = os.path.dirname(os.path.abspath(__file__))
# Ubah nama "best.pt" dan "model_kedua.pt" sesuai dengan nama file model Anda
model_1_path = os.path.join(script_dir, "..", "models", "best.pt")
model_2_path = os.path.join(script_dir, "..", "models", "model_kedua.pt") # Ganti dengan model lain

# Pastikan kedua model exist
print("Loading Models...")
try:
    model1 = YOLO(model_1_path)
except Exception as e:
    print(f"Gagal memuat Model 1: {e}")

try:
    model2 = YOLO(model_2_path)
except Exception as e:
    print(f"Gagal memuat Model 2: {e}")

# 2. Buka kamera webcam
cap = cv2.VideoCapture(0)

# Atur tingkat confidence
conf_level = 0.5

print("Memulai kamera untuk perbandingan... Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca dari kamera.")
        break

    # -- Inferensi Model 1 --
    results1 = model1(frame, conf=conf_level, verbose=False) # verbose=False agar terminal tidak terlalu penuh
    frame1 = results1[0].plot()
    
    # Tambahkan teks label di layar Model 1
    cv2.putText(frame1, "Model 1 (best.pt)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # -- Inferensi Model 2 --
    # Jika Model 2 gagal diload, kita gunakan frame kosong atau asli
    if 'model2' in locals():
        results2 = model2(frame, conf=conf_level, verbose=False)
        frame2 = results2[0].plot()
    else:
        frame2 = frame.copy()
        cv2.putText(frame2, "Model 2 Tidak Ditemukan", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
    # Tambahkan teks label di layar Model 2
    cv2.putText(frame2, "Model 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Gabungkan kedua frame secara horizontal (bersebelahan)
    # Pastikan ukuran kedua frame sama, secara default kamera akan mengeluarkan resolusi sama
    combined_frame = cv2.hconcat([frame1, frame2])
    
    # Supaya tidak terlalu panjang di layar, kita bisa melakukan resize menjadi 70% dari aslinya
    scale_percent = 70
    width = int(combined_frame.shape[1] * scale_percent / 100)
    height = int(combined_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(combined_frame, dim, interpolation=cv2.INTER_AREA)

    # 4. Tampilkan hasil gabungan
    cv2.imshow("YOLOv8 Model Comparison", resized_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()
