from ultralytics import YOLO

# Try to load a model (downloads if not found locally)
model = YOLO("yolov8n.pt")

print("✅ Model loaded successfully")
print("Model names:", model.names)  # prints all 80 COCO classes
