# AI-Powered Runway FOD Detection & Safety Alert System

An end-to-end computer vision prototype for detecting Foreign Object Debris (FOD) on airport runway/taxiway surfaces using deep learning and object detection.

## Project Overview

Foreign Object Debris (FOD) such as nails, bolts, nuts, tools, metal parts, rocks, and other objects can create serious safety risks in airport operations.

This project builds an AI-based prototype that:

- Detects FOD objects in runway/taxiway images
- Localizes objects using bounding boxes
- Predicts the FOD class
- Returns confidence scores
- Supports configurable confidence thresholds
- Provides an interactive FastAPI web interface
- Supports sample-image inference
- Can be containerized and deployed using Docker

> This is a research/portfolio prototype and is not a certified aviation safety system.

---

## Architecture

FOD-A Dataset
↓
Data Validation & Ingestion
↓
EDA
↓
Annotation Analysis
↓
Image Preprocessing
↓
CNN Baseline
↓
YOLO Object Detection
↓
Model Training
↓
Evaluation & Error Analysis
↓
Saved YOLO Model
↓
FastAPI
↓
Interactive Web UI
↓
Docker

---

## Dataset

The project uses the FOD-A dataset in Pascal VOC format.

The dataset contains runway/taxiway images with object-level bounding-box annotations.

### FOD classes

The processed dataset contains 31 FOD classes:

- AdjustableClamp
- AdjustableWrench
- Battery
- Bolt
- BoltNutSet
- BoltWasher
- ClampPart
- Cutter
- FuelCap
- Hammer
- Hose
- Label
- LuggagePart
- LuggageTag
- MetalPart
- MetalSheet
- Nail
- Nut
- PaintChip
- Pen
- PlasticPart
- Pliers
- Rock
- Screw
- Screwdriver
- SodaCan
- Tape
- Washer
- Wire
- Wood
- Wrench

---

## Project Structure

```text
AI Runway FOD Detection/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── artifacts/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
│examples/
│  └── sample_images/
│
├── models/
│   └── fod_yolo_baseline_best.pt
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Annotation_Analysis.ipynb
│   ├── 03_Image_Preprocessing.ipynb
│   ├── 04_CNN_Baseline.ipynb
│   ├── 05_Object_Detection.ipynb
│   └── 06_Evaluation_Error_Analysis.ipynb
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── tests/
│   └── test_project.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
├── run.py
├── prepare_samples.py
└── README.md
