# Containerized microservices for vision and learning tasks

1. YOLOv8 server-client
2. Image dataset visualization using fiftyone
3. Scripts for creating videos from image frames
4. Scripts to stich images to create panoramic images


## Prerequisites
- **Docker**: Ensure Docker Engine is running

## 1. Object Detection using YOLO v8
#### Build the images
`docker build -f Dockerfile.main -t main-application .`  
`docker build -f Dockerfile.yolo-v8-server -t yolo-v8-server .`

#### Run the server-client containers
`docker run --rm -p 8000:8000 yolo-v8-inference-server:latest`  
`docker run --rm -v ./data:/data main-application:latest`

#### [OPTIONAL] (From host) Send a custom request from the command line
E.g. `curl -X POST -F "file=@fruit_1.jpg" http://localhost:8000/predict`


## 2. Image Dataset Visualization using fiftyone

#### Build server image
`docker build -f Dockerfile.fo-server-vis -t fifty-one-vis .`

#### Run server container
`docker run -it --rm -p 5151:5151 fifty-one-vis`

#### Access UI
In a local browser, search for localhost:5151
