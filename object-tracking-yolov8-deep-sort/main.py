import cv2
from ultralytics import YOLO
from tracker import Tracker
import random
import os
import sys
import numpy as np

# Check if the video file name is provided as an argument
if len(sys.argv) < 2:
    raise Exception("Error: Please provide the video file name as a command-line argument.")

# Get the video file name from the argument
video_file_name = sys.argv[1]

# Construct the full relative path to the video
video_folder = "object-tracking-yolov8-deep-sort/input_videos"
video_path = os.path.join(video_folder, video_file_name)

# Output video path in the 'output_video' folder
output_folder = "object-tracking-yolov8-deep-sort/output_video"
os.makedirs(output_folder, exist_ok=True)

# Define the output video path with the same name as the input video
video_out_path = os.path.join(output_folder, f"out_{video_file_name}")

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if video capture is opened successfully
if not cap.isOpened():
    raise Exception(f"Error: Unable to open video file {video_path}")

# Hardcode the frame rate for the output video
output_fps = 30  # Set the desired frame rate here

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer with the hardcoded frame rate
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (frame_width, frame_height), isColor=False)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize Tracker
tracker = Tracker()

# Define random colors for bounding boxes
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = gray_frame[..., np.newaxis]  # Add channel dimension to match model input

    # If your YOLO model requires RGB input, convert grayscale to RGB
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # YOLO model performs detection on the RGB frame
    results = model(rgb_frame)

    detections = []
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)
            # Filter out non-human detections
            if class_id == 0:  # Assuming 0 is the class ID for humans
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                score = float(score)
                detections.append([x1, y1, x2, y2, score])

    # Update tracker with detections
    tracker.update(frame, detections)

    # Draw bounding boxes and track IDs on the original frame
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id

        # Draw bounding box with a random color for each track ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id % len(colors)], 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)

    # Write the processed frame to the output video (in grayscale)
    cap_out.write(gray_frame)

    # Display the processed frame
    cv2.imshow('Video', gray_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video resources
cap.release()
cap_out.release()
cv2.destroyAllWindows()
