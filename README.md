

# **YOLO + DeepSORT Object Tracking**

This project implements **real-time multi-object tracking** using **YOLOv8** for detection and **DeepSORT** for assigning unique IDs to objects across video frames.
Works smoothly on **macOS** with OpenCV display support.

---

## 🚀 Features

* YOLOv8 object detection
* DeepSORT tracking with unique IDs
* Tracks multiple objects (person, car, truck, etc.)
* Supports video files and webcam
* macOS compatible

---

## 📌 Requirements

Install dependencies:

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
```

---

## ▶️ Run

```bash
python yolo_deepsort_mac.py
```

To use webcam:

```python
VIDEO_SOURCE = 0
```

---



## ✨ Output

The script displays a video window showing objects with labels like:

```
person ID 1
car ID 3
```

Each object keeps the **same ID** across frames.

-
