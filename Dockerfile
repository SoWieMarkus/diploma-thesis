# --- Build Frontend stage ---
FROM node:24.19.0-alpine AS builder

WORKDIR /usr/src/app

COPY ./visualizer .

# Install dependencies and build all packages
RUN npm install
RUN npm run build

FROM python:3.14-slim

# Install system dependencies # Install system dependencies needed for building Python packages and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/app/dist/frontend/browser ./visualizer
COPY ./gaze_estimator ./gaze_estimator
COPY ./requirements.txt ./requirements.txt
COPY ./demo.py .
COPY ./evaluate.py .
COPY ./main.py .
COPY ./visualizer.py .
COPY ./data/assets ./data/assets
COPY ./entrypoint.sh .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

