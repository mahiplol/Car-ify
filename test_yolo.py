from ultralytics import YOLO
import cv2

# Load YOLOv11 (nano version)
model = YOLO("yolo11n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, imgsz=640, conf=0.25)

    # Draw boxes on frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv11 - Car Detection", annotated_frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
