# Model Training Pipeline

Folder `pipeline/` ini didedikasikan secara khusus untuk proses pembuatan, pengembangan (development), dan *training* model *machine learning*. Segala alur eksperimen, pemrosesan awal dataset, serta pencarian algoritma atau hyperparameter terbaik sebaiknya dilakukan di folder ini agar rapi dan tidak mengotori *source code* utama proyek.

---

## 📄 File Utama

- **`yolov_roboflow_training_template.ipynb`**  
  File ini adalah notebook Jupyter interaktif yang berfungsi sebagai *template* (kerangka kerja) dari awal hingga akhir (*end-to-end*) untuk melatih (training) model YOLO. Sesuai dengan namanya, kerangka ini sudah diintegrasikan dengan platform anotasi gambar **Roboflow**.

### 🔄 Alur Kerja (Workflow) pada Notebook:
Secara berurutan, sel-sel (cells) di dalam notebook tersebut bertugas untuk:
1. **Setup Lingkungan:** Menginstal *library* Ultralytics YOLO dan modul pendukung lainnya.
2. **Import Dataset (Roboflow):** Meminta API Key Roboflow Anda untuk mengunduh dataset gambar dan label yang sudah Anda anotasikan sebelumnya tanpa harus mengunduhnya secara manual ke komputer.
3. **Training Model:** Menjalankan perintah pelatihan `yolo train` secara mandiri untuk mengajari AI mendeteksi objek Anda dengan menyesuaikan parameter seperti *epochs* dan *batch sizes*.
4. **Visualisasi Validasi:** Memantau kemajuan model yang sedang dilatih lewat grafik akurasi (Precision, Recall, mAP).
5. **Penyimpanan:** Memvalidasi file bobot akhir (`best.pt`) yang sudah ter-*training* dan siap dipakai.

---

## 🚀 Panduan Penggunaan
Jika Anda ingin melatih model baru, merevisi model lama, atau mengganti dataset:

1. **Jalankan Notebook:** Anda dapat membuka dan menjalankan `.ipynb` ini menggunakan **Jupyter Notebook / VS Code** di komputer Anda sendiri (jika memiliki GPU/VGA seperti NVIDIA dengan CUDA), atau unggah dan jalankan secara gratis di **Google Colab** untuk mempercepat waktu *training*.
2. **Koneksikan Dataset:** Saat diminta, masukkan kode *snippet* dari dataset Roboflow Anda.
3. **Ambil Hasil Model:** Setelah *training* selesai, sistem akan menghasilkan folder `runs/detect/train/weights/` secara otomatis.
4. **Pindahkan Model:** Ambil file `best.pt` dari dalam folder hasil tersebut, lalu *copy* dan letakkan di dalam folder `models/` utama milik proyek ini.
5. **Uji Coba Langsung:** Anda kini dapat langsung mengetes kecerdasan model buatan Anda di dunia nyata dengan menggunakan script `run_model.py` yang ada di folder `scripts/`.
