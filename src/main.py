import cv2
import requests
import numpy as np
from io import BytesIO
import os

# Server URL for YOLOv8 inference (adjust this based on your actual server)
YOLO_SERVER_URL = 'http://host.docker.internal:8000/predict'

# Set up video capture for the example.mp4 file
cap = cv2.VideoCapture('/data/example.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video properties (width, height, and frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up the video writer to save the output
out = cv2.VideoWriter('/data/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame in the video
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Send the image to the YOLOv8 server for inference
    try:
        response = requests.post(YOLO_SERVER_URL, files={"file": ("frame.jpg", BytesIO(img_bytes), "image/jpeg")})

        if response.status_code == 200:
            # Print the entire response to inspect its structure
            print("Response from YOLO server:", response.text)  # Debugging output

            # Assuming YOLOv8 server responds with a JSON containing bounding boxes
            results = response.json()

            # Check if 'predictions' exist and loop through the detected objects to draw bounding boxes
            if 'predictions' in results:
                for obj in results['predictions']:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = obj['bbox']
                    confidence = obj['confidence']
                    class_id = obj['class_id']  # Use 'class_id' instead of 'class'

                    # Print object details for debugging
                    print(f"Detected: class_id {class_id} with confidence {confidence:.2f}, bbox: ({x1},{y1}) -> ({x2},{y2})")

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {class_id} {confidence:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("No predictions found in the response.")
        else:
            print(f"Error: Failed to get response from YOLO server, status code {response.status_code}")

    except Exception as e:
        print(f"Error during inference request: {e}")

    # Write the processed frame to the output video
    out.write(frame)

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

# Release resources
cap.release()
out.release()
print("Processing complete. Output saved to '/data/output_video.mp4'.")
