# 🚗 Real-Time Car Recognition AI

A camera-powered system that recognizes cars in real time, identifies their make, model, and year using YOLOv11, and generates live AI "car dossiers" with expressive summaries powered by OpenAI.

---

## Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Usage Controls](#usage-controls)
- [Example Output](#example-output)
- [Use Cases](#use-cases)
- [Contributors](#contributors)

---

## Introduction

This project is a **real-time AI car recognition system** that combines computer vision and generative AI. Using a standard webcam or mobile camera, the system:

- Detects vehicles in real time using **YOLOv11**
- Classifies them into **make, model, and year** using a fine-tuned YOLOv11-cls model
- Fetches **technical specifications** (engine type, drivetrain, etc.) from a local JSON dataset
- Generates **live AI summaries** ("car dossiers") using **OpenAI's GPT-4o-mini**
- Displays everything in a real-time overlay interface

The system transforms a simple camera feed into an intelligent car recognition companion within seconds.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Detection & Classification** | Ultralytics YOLOv11, PyTorch |
| **Backend API** | FastAPI, OpenAI SDK |
| **Data Handling** | JSON (car specs, labels) |
| **Visualization** | OpenCV (real-time video overlay) |
| **Hardware** | CPU and GPU (CUDA optional) |

---

## System Architecture
Camera Feed
│
▼
[ YOLOv11 Object Detector ]
│
▼
[ YOLOv11-cls Classifier ]
│
├──→ Local JSON lookup → specs (engine, drivetrain, etc.)
│
└──→ FastAPI → OpenAI API → expressive summary
│
▼
[ Real-time Overlay (OpenCV) ]
│
├── Green boxes → recognized cars
└── L-key → generates AI dossier on demand

text

---

## Folder Structure
car-vision-test/
│
├── data/
│ ├── raw_images/ # ~60,000 car images
│ └── car_cls_triplet/ # Organized YOLO classification dataset
│
├── tools/
│ ├── prepare_cls_dataset.py # Builds YOLOv11 classification dataset
│ └── _analysis/
│ ├── class_specs.json # Car specs extracted from filenames
│ └── metadata.csv # Dataset summary
│
├── models/
│ └── best.pt # Trained YOLOv11-cls weights
│
├── api/
│ └── app.py # FastAPI backend (OpenAI integration)
│
├── detect_and_classify.py # Main real-time application
├── test_yolo.py # YOLOv11 webcam test
├── capture_crops.py # Optional manual cropper
├── requirements.txt # Dependencies
└── README.md

text

---

## How It Works

### Dataset Preparation
- 60,000+ car images parsed and grouped by make, model, year
- Filenames encode metadata like MSRP, engine type, drivetrain
- `prepare_cls_dataset.py` automatically organizes training/validation folders

### Model Training
bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-cls.pt').train(data='data/car_cls_triplet', epochs=30, imgsz=224, batch=16, device='cuda')"
FastAPI Backend
Endpoint: /dossier

Input: { "label": "lexus_es_2013" }

Output: JSON with specs, estimated price, and OpenAI-generated summary

Real-Time Camera
Runs YOLOv11 detection & tracking

After 2 seconds steady view, classifies the car

Press L → calls OpenAI API for descriptive summary

Press P → prints full dossier to console

Installation & Setup
Clone and enter project

bash
git clone https://github.com/yourusername/car-vision-test.git
cd car-vision-test
Create virtual environment

bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# or
source .venv/bin/activate        # macOS/Linux
Install dependencies

bash
pip install ultralytics fastapi uvicorn opencv-python pillow numpy pydantic timm openai
Set OpenAI API key

bash
$env:OPENAI_API_KEY = "sk-your-real-key"
Add your trained model
Place your trained classification model inside:

text
models/best.pt
Running the App
Start the backend

bash
uvicorn api.app:app --reload --port 8000
Start the detection/classification camera

bash
python detect_and_classify.py --det yolo11n.pt --cls models/best.pt --label_map data/car_cls_triplet/label_map.json --specs tools/_analysis/class_specs.json --source 0 --hold_secs 2.0
Usage Controls
Key	Action
ESC	Quit program
L	Fetch and overlay AI-generated car summary
P	Print full dossier (title, specs, summary)
(Auto)	Classifies car after ~2 seconds of steady detection
Example Output
In-App Overlay
text
Lexus ES 2013  (0.88)
MSRP: $36,000   HP@RPM: 268@6200   FWD (incomplete due to lack of reliable data)
A serene luxury sedan praised for its whisper-quiet ride and timeless reliability.
Console Output (after pressing P)
text
=== Dossier ===
Label:     lexus_es_2013
Title:     Lexus ES 2013
Price:     $10,000–$13,000 (rough estimate)(incomplete due to lack of reliable data)
Condition: Good
Summary:   A serene luxury sedan praised for its whisper-quiet ride and timeless reliability.
Specs:     {'drivetrain': 'FWD', 'engine': 'V6', 'body': 'Sedan'}
=============
Use Cases
Domain	Use Case
Car Shows / Auctions	Visitors point camera → instant car summaries
Dealership Apps	Real-time inventory recognition + AI-driven information
Education / Enthusiasts	Identify classic models and display technical details
Fleet Management	Validate vehicle types using live camera input
Automotive Research	Rapid vehicle identification and data collection
Contributors
Mahip
Keito
Kabir

Summary
This project demonstrates a seamless fusion of computer vision and generative AI:

Detect → Identify → Describe

In seconds, it transforms a camera feed into an intelligent car recognition system capable of providing detailed vehicle information and AI-generated summaries.

Last updated: November 2025
