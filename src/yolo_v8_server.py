from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load the YOLOv8 model (adjust path as needed)
model = YOLO("yolov8s.pt")

@app.get("/")
def root():
    return {"message": "Hello from YOLOv8 on macOS (CPU Docker)"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read the image file from the client
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image into a NumPy array
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Run inference using YOLOv8 model
    results = model.predict(source=image, conf=0.25)

    # Prepare the detections response
    detections = []
    for result in results:
        # Extract the bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates: [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        # Append the results in the expected format
        for box, score, cls_id in zip(boxes, scores, classes):
            detections.append({
                "bbox": [float(x) for x in box],  # Bounding box coordinates as floats
                "confidence": float(score),       # Confidence score
                "class_id": int(cls_id)           # Class ID
            })

    # Return the predictions in a format that matches the client-side code
    return JSONResponse(content={"predictions": detections})
