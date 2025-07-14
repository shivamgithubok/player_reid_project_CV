from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=15)

    def update(self, detections, frame, frame_idx):
        formatted = [
            [
                [d['bbox'][0], d['bbox'][1], d['bbox'][2] - d['bbox'][0], d['bbox'][3] - d['bbox'][1]],  # [x, y, w, h]
                d['confidence'],
                'player'
            ]
            for d in detections if d['class'] == 0
        ]
        if not formatted:
            return []
        tracks = self.tracker.update_tracks(formatted, frame=frame)
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            tracked_objects.append({
                'id': track_id,
                'bbox': [int(l), int(t), int(l + w), int(t + h)]
            })
        return tracked_objects