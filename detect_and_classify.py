# detect_and_classify.py
# Vehicle-only detection, hard gates, crop-only classification, hold timer,
# and L-key "Learn" dossier fetch via FastAPI with visible chips/summary band.

import time
import json
import argparse
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO
from PIL import Image

# ---------------------------
# Config: detection & classify gates
# ---------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
DET_CONF = 0.60                  # detector confidence threshold
DET_IOU  = 0.60                  # detector NMS IoU
MIN_WH   = 64                    # min box width/height in px to classify
MIN_AREA_FRAC = 0.02             # min box area as fraction of frame (e.g., 2%)
CLS_CONF_MIN = 0.60              # classifier top-1 confidence must be >= this

# Dossier API (your FastAPI server from api/app.py)
DOSSIER_API = "http://127.0.0.1:8000/dossier"

# ---------------------------
# Helpers
# ---------------------------
def load_json(p: str | Path) -> dict:
    p = Path(p)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def open_capture(src_arg: str):
    # Parse camera index or file path; try DirectShow on Windows if needed
    if src_arg.isdigit():
        idx = int(src_arg)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        return cap
    return cv2.VideoCapture(src_arg)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="yolo11n.pt", help="YOLO detection model (COCO)")
    ap.add_argument("--cls", required=True, help="Path to trained classification best.pt")
    ap.add_argument("--label_map", default="data/car_cls_triplet/label_map.json")
    ap.add_argument("--specs", default="tools/_analysis/class_specs.json")
    ap.add_argument("--source", default="0", help="Camera index or video file path")
    ap.add_argument("--hold_secs", type=float, default=1.5, help="Seconds to hold a stable track before classifying")
    ap.add_argument("--imgsz", type=int, default=224, help="Classifier input size")
    args = ap.parse_args()

    # Load models
    det = YOLO(args.det)
    cls = YOLO(args.cls)

    # Load metadata
    label_map = load_json(args.label_map)
    specs_map = load_json(args.specs)

    # Video source
    cap = open_capture(args.source)
    if not cap or not cap.isOpened():
        print(f"[ERR] Cannot open source: {args.source}")
        return

    print("[INFO] Running... 'L' = Learn (fetch dossier), 'P' = print dossier. ESC to quit.")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Per-track state: { track_id: {first_seen, label, score, dossier, dossier_llm} }
    track_state: dict[int, dict] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        H, W = frame.shape[:2]

        # Vehicle-only tracking with thresholds
        results = det.track(
            frame,
            persist=True,
            verbose=False,
            classes=VEHICLE_CLASSES,
            conf=DET_CONF,
            iou=DET_IOU
        )

        # If no detections, clear state & show raw frame
        if (not results) or all(getattr(r, "boxes", None) is None or len(r.boxes) == 0 for r in results):
            track_state.clear()
            cv2.imshow("Detect + Classify (Make_Model_Year)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('p'), ord('P')):
                print("[DEBUG] No tracks.")
            # 'L' ignored when no targets
            continue

        for r in results:
            if not getattr(r, "boxes", None) or r.boxes is None or len(r.boxes) == 0:
                continue

            ids  = getattr(r.boxes, "id",   None)  # track ids
            clss = getattr(r.boxes, "cls",  None)  # detector classes
            conf = getattr(r.boxes, "conf", None)  # detector confidences

            for i, box in enumerate(r.boxes.xyxy):
                # Double-safety: ensure detector class is a vehicle
                cls_id = int(clss[i].item()) if clss is not None and len(clss) > i else -1
                if cls_id not in VEHICLE_CLASSES:
                    continue

                det_conf = float(conf[i].item()) if conf is not None and len(conf) > i else 0.0
                if det_conf < DET_CONF:
                    continue

                # Box coords and hard gates
                x1, y1, x2, y2 = map(int, box.tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                w, h = x2 - x1, y2 - y1
                area_frac = (w * h) / float(W * H + 1e-9)
                if w < MIN_WH or h < MIN_WH or area_frac < MIN_AREA_FRAC:
                    continue

                # Track id or pseudo-id from coarse coords
                tid = None
                if ids is not None and len(ids) > i and ids[i] is not None:
                    try:
                        tid = int(ids[i].item())
                    except Exception:
                        tid = None
                if tid is None:
                    tid = hash((x1 // 16, y1 // 16, x2 // 16, y2 // 16))

                st = track_state.get(tid)
                if st is None:
                    st = {"first_seen": now, "label": None, "score": 0.0,
                          "dossier": None, "dossier_llm": None}
                    track_state[tid] = st

                # Draw box (yellow if waiting, green if labeled)
                color = (0, 255, 0) if st["label"] else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Classify only after hold_secs and only on the crop
                if (st["label"] is None) and ((now - st["first_seen"]) >= args.hold_secs):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    res = cls.predict(crop, imgsz=args.imgsz, verbose=False)[0]
                    p = res.probs
                    idx = int(p.top1)
                    conf_top1 = float(getattr(p, "top1conf", None) or p.data[idx])

                    if conf_top1 < CLS_CONF_MIN:
                        cv2.putText(frame, f"Unknown vehicle ({conf_top1:.2f})",
                                    (x1, max(10, y1 - 8)), font, 0.55, (0, 255, 255), 2)
                        continue

                    lab = cls.names[idx]  # e.g., 'acura_ilx_2013'
                    # Resolve human-readable fields
                    disp = label_map.get(lab)
                    if disp:
                        make, model, year = disp.get("make", "?"), disp.get("model", "?"), disp.get("year", "?")
                    else:
                        parts = lab.split("_", 2)
                        make = parts[0].title() if len(parts) > 0 else "Unknown"
                        model = parts[1].upper() if len(parts) > 1 else "?"
                        year = parts[2] if len(parts) > 2 else "?"

                    spec = specs_map.get(lab, {})
                    st["label"] = lab
                    st["score"] = conf_top1
                    st["dossier"] = {
                        "make": make, "model": model, "year": year,
                        "msrp": spec.get("MSRP"),
                        "hp_rpm": spec.get("SAEhpRPM"),
                        "displacement": spec.get("Displacement"),
                        "engine": spec.get("EngineType"),
                        "gas_mileage": spec.get("GasMileage"),
                        "drivetrain": spec.get("Drivetrain"),
                        "width_in": spec.get("WidthIn"),
                        "height_in": spec.get("HeightIn"),
                        "length_in": spec.get("LengthIn"),
                        "passengers": spec.get("PassengerCapacity"),
                        "doors": spec.get("PassengerDoors"),
                        "bodystyle": spec.get("BodyStyle"),
                        "conf": conf_top1,
                    }
                    st["dossier_llm"] = None  # ready for Learn (L)

                # Overlay text (top lines)
                if st["label"]:
                    d = st["dossier"] or {}
                    line1 = f"{d.get('make','?')} {d.get('model','?')} {d.get('year','?')}  ({d.get('conf',0.0):.2f})"
                    line2 = f"MSRP:{d.get('msrp','?')}  HP@RPM:{d.get('hp_rpm','?')}  {d.get('drivetrain','')}"
                    cv2.putText(frame, line1, (x1, max(10, y1 - 22)), font, 0.55, (0, 255, 0), 2)
                    cv2.putText(frame, line2, (x1, max(10, y1 - 4)),  font, 0.5,  (0, 255, 0), 1)

                    # LLM chip & summary band
                    llm = st.get("dossier_llm")
                    if isinstance(llm, dict):
                        # Chip above box: "[Fetching...]" or "[Learned]"
                        chip = "[Learned]" if llm.get("summary") else "[Fetching...]"
                        (tw, th), _ = cv2.getTextSize(chip, font, 0.5, 1)
                        bx, by = x1, max(0, y1 - 40)
                        cv2.rectangle(frame, (bx, by - th - 6), (bx + tw + 10, by + 4), (0, 0, 0), -1)
                        cv2.putText(frame, chip, (bx + 5, by - 4), font, 0.5, (255, 255, 255), 1)

                        # Summary band below box when available
                        if llm.get("summary"):
                            summary_line = llm["summary"]
                            summary_line = (summary_line[:90] + "…") if len(summary_line) > 92 else summary_line
                            base_y = min(H - 8, y2 + 22)
                            (sw, sh), _ = cv2.getTextSize(summary_line, font, 0.48, 1)
                            cv2.rectangle(frame, (x1, base_y - sh - 8), (min(W - 1, x1 + sw + 10), base_y + 4), (0, 0, 0), -1)
                            cv2.putText(frame, summary_line, (x1 + 5, base_y - 4), font, 0.48, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "Hold steady...", (x1, max(10, y1 - 6)),
                                font, 0.5, (0, 255, 255), 1)

        # Show & handle keys (ESC/L/P)
        cv2.imshow("Detect + Classify (Make_Model_Year)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('l'), ord('L')):
            # Find first confirmed track without LLM dossier and fetch once
            for tid, st in track_state.items():
                if st.get("label") and st.get("dossier") and st.get("dossier_llm") is None:
                    try:
                        payload = {"label": st["label"]}
                        resp = requests.post(DOSSIER_API, json=payload, timeout=4)
                        if resp.ok:
                            st["dossier_llm"] = resp.json()
                            print(f"[INFO] Dossier fetched for {st['label']}")
                        else:
                            st["dossier_llm"] = {"summary": "Dossier service unavailable."}
                    except Exception as e:
                        print("[ERR] Dossier request failed:", e)
                        st["dossier_llm"] = {"summary": "Request failed."}
                    break  # only one per keypress
        elif key in (ord('p'), ord('P')):
            # Print the first confirmed track's dossier JSON for debug
            printed = False
            for tid, st in track_state.items():
                if st.get("label") and st.get("dossier"):
                    print("[DEBUG dossier]", st["label"], st.get("dossier_llm"))
                    printed = True
                    break
            if not printed:
                print("[DEBUG] No confirmed tracks to print.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
