import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, out_features=14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(image_rgb).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().detach().numpy()
        original_height, original_width = frame.shape[:2]
        keypoints[::2] *= original_width / 224.0
        keypoints[1::2] *= original_height / 224.0

        return keypoints
    
    def draw_keypoints(self, frame, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return frame
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames