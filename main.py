from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

model = YOLO("yolov8s.pt")

@app.get("/")
def root():
    return {"message": "Hello from YOLOv8 on macOS (CPU Docker)"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = model.predict(source=image, conf=0.25)

    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

    for box, score, cls_id in zip(boxes, scores, classes):
        detections.append({
            "bbox": [float(x) for x in box],
            "score": float(score),
            "class_id": int(cls_id)
        })

    return JSONResponse(content={"detections": detections})
