import torch
import cv2
from PIL import Image
import numpy as np
import time

# Load YOLOv5 model (use yolov5s for faster processing)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a named window
cv2.namedWindow('YOLOv5 Object Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('YOLOv5 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to improve processing speed
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # Perform object detection
    results = model(img)

    # Extract detected objects and their bounding boxes
    detected_objects = results.xyxy[0]  # Extract the first set of detected objects

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in detected_objects:
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS
    fps = 1 / (time.time() - start_time)

    # Display FPS on the frame
    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
