import cv2 
import os
from ultralytics import YOLO

# Get absolute path to the models directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "best.pt")

model = YOLO(model_path)

# Buka kamera webcam (index 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca dari kamera.")
        break

    # Lakukan prediksi pada setiap frame
    # Anda bisa mengatur 'conf' (confidence level). 0.5 artinya 50%.
    # Silakan ubah nilainya (contoh: 0.25, 0.7, dll) sesuai kebutuhan Anda.
    results = model(frame, conf=0.5)
    
    # Ambil frame yang sudah digambar kotak deteksi (bounding box)
    annotated_frame = results[0].plot()
    
    # Tampilkan frame di window GUI
    cv2.imshow("YOLOv8 Webcam Inference", annotated_frame)
    
    # Tekan 'q' pada keyboard untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources dan tutup window
cap.release()
cv2.destroyAllWindows()