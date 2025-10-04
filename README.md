# Real-time-object-detection-using-yolov8
Implemented a real-time object detection system using the YOLO (You Only Look Once) algorithm. Used a pre-trained YOLOv8 model to detect and label multiple objects in live video feed or images. Made an interface and it detected multiple objects on realtime including cell phone, person, pizza, couch, bed, dog, and cats etc. 

Prerequisites

Before running the code, ensure you have the following installed:

Python 3.8+ (Recommended: 3.9 or 3.10)

pip package manager

A working webcam or external USB camera

📦 Installation

Clone or download this repository

git clone https://github.com/your-username/yolov8-webcam-detection.git
cd yolov8-webcam-detection


Install dependencies

pip install ultralytics opencv-python


Download YOLOv8n model weights

You can download yolov8n.pt from the official Ultralytics release:
YOLOv8 pretrained models

Place it in your working directory or update the path in the script.

▶️ Usage

Run the detection script:

python realtime_yolo.py

