import cv2

def draw_boxes(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj['id']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def save_video(frames, path, fps=30):
    if not frames:
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()