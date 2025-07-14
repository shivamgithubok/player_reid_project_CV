from ultralytics import YOLO
import cv2

MODEL_PATH = 'models/best.pt'

model = YOLO(MODEL_PATH)
model.fuse()  # Optional: for speed

def run_detection(frame):
    results = model(frame)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls
            })
    return detections