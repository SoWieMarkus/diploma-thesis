# Diploma Thesis: Conception and Evaluation of a Smartphone-based Eye-Tracking Solution for Use in Bicycle Rides with High Robustness against Changing Environmental Conditions

This repository contains the complete implementation and documentation of my diploma thesis submitted on December 4th, 2023, at the Technical University of Dresden (Technische Universität Dresden). The project develops a smartphone-based eye-tracking system designed specifically for cyclists, contributing to safer navigation app usage during bicycle rides.

## Screenshots

## Content

## 🚀 Try it yourself

### Prerequisites

- [Download Python](https://www.python.org/downloads/)
- [Download Node.js](https://nodejs.org/)
- Install Angular CLI with `npm install -g @angular/cli`

### Set up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

- create a .venv
- install requirements.txt
- run `python demo.py`

### Run demo.py

The demo script supports various configuration options:

```bash

python demo.py

# Use different face detector (mediapipe or dlib)
python demo.py --face_detector mediapipe

# Use different gaze estimation model
python demo.py --gaze_detector eth-xgaze
# Available options: eth-xgaze, gaze360, mpiifacegaze, head-rotation

# Use specific camera (if you have multiple cameras)
python demo.py --cam 1

# Customize visualization
python demo.py --draw_important_landmarks    # Show only key facial landmarks
python demo.py --skip_landmarks             # Hide facial landmarks
python demo.py --skip_gaze_vector          # Hide gaze direction vector
```

## Abstract

The climate crisis represents one of the greatest challenges of our time, with urban traffic being a major contributor through high CO2 and particulate matter emissions. To address this issue, the city of Hamburg initiated the PrioBike project, which aims to make cycling more attractive and revolutionary through optimized navigation apps that provide speed recommendations based on traffic light data.

However, smartphone usage while cycling poses significant safety risks due to potential distractions. This thesis investigates the feasibility of smartphone-based eye-tracking solutions to monitor when cyclists look at their devices, enabling the identification of distracting app interfaces and evaluation of alternative interaction methods. The work aimed to develop and evaluate a robust smartphone-based eye-tracking system for cyclists that operates using only the front-facing camera, without additional hardware or calibration requirements, while maintaining high robustness against changing environmental conditions.

Various gaze estimation approaches were investigated, including deep learning appearance-based methods and head rotation-based techniques. Face detection was performed using MediaPipe and dlib, followed by 3D gaze direction estimation using convolutional neural networks trained on ETH-XGaze, Gaze360, and MPIIFaceGaze datasets. The system determines whether cyclists are looking at their smartphone display by modeling a detection zone on the camera plane.

The best performance was achieved using MediaPipe for face detection combined with ETH-XGaze for gaze estimation, reaching an F1-score of 0.928. The system successfully compensated for normal environmental influences including vibrations and varying lighting conditions. However, challenges were identified with sunglasses, night recordings, and extreme head rotations.

The developed system demonstrates that smartphone-based eye-tracking for cyclists is feasible and can effectively determine when users look at their devices. The solution is robust against typical cycling conditions and provides a cost-effective approach for evaluating app interfaces and interaction methods in real-world cycling scenarios. This work contributes significantly to the safe usage of navigation apps while cycling and provides a foundation for future studies within the PrioBike project.
