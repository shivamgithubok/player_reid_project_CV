from scipy.spatial.distance import cosine

def match_ids(tracks, prev_features, current_features, threshold=0.5):
    matched_tracks = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        best_match = tid
        min_dist = float('inf')
        for pid, prev_feat in prev_features.items():
            if tid not in current_features:
                continue
            dist = cosine(prev_feat, current_features[tid])
            if dist < threshold and dist < min_dist:
                best_match = pid
                min_dist = dist
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        matched_tracks.append({
            'id': best_match,
            'bbox': [x1, y1, x2, y2]
        })
    return matched_tracks