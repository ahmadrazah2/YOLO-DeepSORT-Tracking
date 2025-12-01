from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # small and fast

results = model.predict(
    source="cctv.mp4",  # or 0 for webcam
    save=True,
    conf=0.5,
    show=True
)

