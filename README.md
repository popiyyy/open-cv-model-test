# YOLOv8 OpenCV Inference Project

This project contains a collection of Python scripts to load, evaluate metrics, run real-time inference with a webcam, and compare the performance of object detection models (YOLOv8) using an OpenCV-based approach.

---

## 📁 Directory Structure

```text
.
├── models/             # Folder to place model files (.pt), e.g., best.pt, model2.pt
├── scripts/
│   ├── run_model.py    # Script for live object detection via webcam using 1 model
│   ├── compare_model.py# Script to visually compare 2 models (side-by-side)
│   └── cek_metric.py   # Script to check evaluation metrics (mAP, Precision, Recall)
├── requirements.txt    # List of required Python module dependencies
└── .gitignore          # File to exclude the environment and large files from Git
```

---

## 🛠️ Preparation & Installation

1. **Create and Activate a Virtual Environment**  
   It is recommended to run the code in a *virtual environment* to keep it clean:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # For Windows
   ```

2. **Install Dependencies**  
   Run the *requirements* file to install YOLO (Ultralytics), OpenCV, and others:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the YOLO Model**  
   Place your trained YOLO model file (usually with a `.pt` extension) into the `models/` folder.

---

## 🚀 Script Usage

### 1. Real-time Detection (Single Webcam)
Use your computer's webcam to detect objects.
```bash
python scripts/run_model.py
```
> **Tip:** You can stop the camera session at any time by pressing the `q` key on the keyboard. If you want to change the detection confidence threshold, edit the `conf=0.5` value inside the script.

### 2. Compare 2 Models Simultaneously (Side-by-Side)
If you have several iterations/variations of a model (e.g., `best.pt` and `model2.pt`) and are unsure which one performs better:
```bash
python scripts/compare_model.py
```
This script will open the webcam and process the same frames to feed them into both models, then combine them on one screen side-by-side (*split mode*).

### 3. Evaluate Accuracy Metrics
Want to see detailed accuracy metrics like mAP50, Precision, and Recall?
1. Edit the `scripts/cek_metric.py` file. Uncomment the *try-catch statement*.
2. Point the dataset *path* marked with `data='path/to/dataset.yaml'`. *(You need the data.yaml file left over from training the model used).*
3. Run:
```bash
python scripts/cek_metric.py
```

---

## 📝 Important Notes
- In the `.gitignore` configuration, the `models/` folder and its contents will be ignored by Git. This is done because AI model files (`.pt`) are relatively large and might exceed the upload size limit of repositories like GitHub. If you still want to include the model, you can adjust the ignore rules in `.gitignore`.
