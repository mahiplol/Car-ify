# capture_crops.py
from ultralytics import YOLO
import cv2, os, time
from datetime import datetime
from pathlib import Path

# Settings
MODEL_PATH = "yolo11n.pt"
SAVE_ROOT = Path("data/crops_raw")  # raw, unlabeled crops
CONF = 0.25
IMG_SIZE = 640

SAVE_ROOT.mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)  # change index if needed
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("[i] Press 's' to save crops from current frame, 'ESC' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break

    res = model(frame, imgsz=IMG_SIZE, conf=CONF)
    p = res[0]
    annotated = p.plot()

    # draw and show
    cv2.imshow("Capture Crops (press 's' to save)", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    # Save all detected car crops when 's' is pressed
    if key == ord('s'):
        boxes = p.boxes
        if boxes is not None and len(boxes) > 0:
            saved = 0
            for b in boxes:
                cls_id = int(b.cls[0].item())
                # COCO class id for 'car' is 2 for YOLOv5/8/11 default; ultralytics maps classes internally.
                # Safer: use p.names[cls_id] and check == 'car'
                label = p.names[cls_id]
                if label != 'car': 
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: 
                    continue
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = SAVE_ROOT / f"car_{ts}.jpg"
                cv2.imwrite(str(out_path), crop)
                saved += 1
            print(f"[+] Saved {saved} car crop(s)")
        else:
            print("[!] No cars detected in this frame")

cap.release()
cv2.destroyAllWindows()
