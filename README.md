# 🐍 Python OJD – Object Detection (YOLOv8 + Roboflow)

This repository contains an end-to-end Object Detection project (OJD) using **Ultralytics YOLOv8**. The goal is to detect **three shape categories**:

* **Square**
* **Circle**
* **Long Object** (rectangular / elongated shape)

The dataset is prepared and annotated using **Roboflow**, and the training + inference pipeline is built in Python.

---

## 🚀 Features

* Train and run YOLOv8 for custom object detection
* Uses dataset from **Roboflow** (auto-generated YAML)
* Supports detecting multiple geometric shapes
* Includes inference script for images, videos, and webcam
* Export model to ONNX / TensorRT if needed

---

## 📂 Project Structure

```
python-ojd/
│
├── data/                 # Roboflow dataset (downloaded automatically)
├── models/               # YOLOv8 models (weights)
├── src/
│   ├── train.py          # Training script
│   ├── predict.py        # Inference script
│   └── utils.py          # Helper functions
│
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🧰 Requirements

* Python 3.8+
* Ultralytics YOLOv8
* Roboflow Python SDK (optional)
* OpenCV (for image/video inference)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics roboflow opencv-python
```

---

## 📥 Download Dataset From Roboflow

Update with your own Roboflow API key and workspace/project details.

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("your-dataset")
dataset = project.version(1).download("yolov8")
```

This will download dataset and YAML to `data/`.

---

## 🏋️ Train YOLOv8 Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy
model.train(data="data/data.yaml", epochs=50, imgsz=640)
```

Training output and weights will be saved in `runs/detect/train/`.

---

## 🔍 Run Object Detection

### Inference on Image

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="sample.jpg", show=True)
```

### Inference on Webcam

```bash
python src/predict.py --source 0
```

---

## 📝 Example Classes

Your dataset should include labels like:

```
square
circle
long_object
```

Ensure these match the annotations in Roboflow.

---

## 📦 Export Model

```python
model.export(format="onnx")
```

Other export formats supported: TensorRT, CoreML, TFLite.

---

## 📊 Results

You can document:

* mAP performance
* Example predictions
* Notes about performance and future improvements

Add images to the repo (example):

```
results/
   ├── prediction1.jpg
   ├── prediction2.jpg
```

---

## 🧪 Future Improvements

* this methode already implement to detect roller bearing conveyor
* the project can detetc bearing abnormal

---

## 📄 License

MIT License

---



