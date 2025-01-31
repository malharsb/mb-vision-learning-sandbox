# Containerized YOLO v8 Inference Server

## Prerequisites

- **Docker**: Ensure Docker Engine is running
- **curl**

## 1. Build the Docker Image
`docker build -t yolo-v8-inference-server .`

## 2. Start and run container
`docker run --rm -p 8000:8000 yolo-v8-inference-server:latest`

### 3. Send a request from the command line
`curl -X POST -F "file=@fruit_1.jpg" http://localhost:8000/predict`
