import torch
import cv2
import sys
import os
from sort import Sort

# 1. Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 2. Initialize the Tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# 3. Get video path from Terminal command
if len(sys.argv) < 2:
    print("Usage: python3 run_tracking.py <path_to_video>")
    sys.exit()

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# --- SAVE FEATURE SETUP ---
video_name = os.path.basename(video_path).split('.')[0]
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# Saves as 'test_video_01_tracked.mp4' etc.
out = cv2.VideoWriter(f'tracked_{video_name}.mp4', fourcc, fps, (width, height))

print(f"Processing: {video_name}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame)
    dets = results.xyxy[0].cpu().numpy()
    person_dets = dets[dets[:, 5] == 0]

    # Update tracker (Kalman + Hungarian)
    tracked_objs = tracker.update(person_dets[:, :5])

    for obj in tracked_objs:
        x1, y1, x2, y2, obj_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Finished: tracked_{video_name}.mp4")