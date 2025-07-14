# main.py
from src.detect import run_detection
from src.tracker import PlayerTracker
from src.utils import draw_boxes, save_video
import cv2

VIDEO_PATH = "videos/15sec_input_720p.mp4"
OUTPUT_PATH = "outputs/tracked_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
tracker = PlayerTracker()

frame_idx = 0
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = run_detection(frame)  # Detect players
    tracked_objects = tracker.update(detections, frame, frame_idx)  # Track & re-identify

    output_frame = draw_boxes(frame.copy(), tracked_objects)
    frames.append(output_frame)

    frame_idx += 1

cap.release()
save_video(frames, OUTPUT_PATH)
print("Tracking complete. Output saved to:", OUTPUT_PATH)