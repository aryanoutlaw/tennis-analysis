from ultralytics import YOLO

model = YOLO("models/yolo5_last.pt")

model.predict("input_videos/input_video.mp4", save=True)