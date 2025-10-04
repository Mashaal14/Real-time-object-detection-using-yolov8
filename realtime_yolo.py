import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8n model (local path where you saved yolov8n.pt)
model = YOLO("E:/DL_internship_task3,4/task3/yolov8n.pt")

# Open webcam (0 = default laptop camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Add FPS text on frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show detection
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
