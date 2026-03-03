"""Quick diagnostic to see what the YOLO model returns."""

import cv2
from ultralytics import YOLO

video_path = "/home/ss/Kirti/lat/video_test/12762044-hd_1920_1080_60fps.mp4"
model_path = "/home/ss/Kirti/lat/models/yolov26nobbnew_merged.pt"

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

print("Testing first 5 frames...\n")
for frame_idx in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.01, verbose=False)
    result = results[0]
    
    print(f"Frame {frame_idx}:")
    print(f"  result.boxes type: {type(result.boxes)}")
    print(f"  result.boxes: {result.boxes}")
    
    if result.boxes is not None:
        print(f"  len(result.boxes): {len(result.boxes)}")
        print(f"  result.boxes.conf: {result.boxes.conf}")
        print(f"  result.boxes.cls: {result.boxes.cls}")
        print(f"  result.boxes.xyxy: {result.boxes.xyxy}")
        
        # Try to access the attributes
        try:
            confs = result.boxes.conf.cpu().numpy()
            print(f"  Extracted confs: {confs}")
        except Exception as e:
            print(f"  ERROR extracting confs: {e}")
    
    annotated = result.plot()
    print(f"  annotated frame shape: {annotated.shape}")
    print()

cap.release()
