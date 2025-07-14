import torch
import torchvision.transforms as T
from torchvision.models import resnet18
import cv2

# Load pretrained ResNet model
model = resnet18(pretrained=True)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(frame, tracks):
    features = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        img = transform(crop).unsqueeze(0)
        with torch.no_grad():
            feat = model(img).squeeze().numpy()
        features[track.track_id] = feat
    return features