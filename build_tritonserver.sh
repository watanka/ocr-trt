#!/bin/bash

requirements=(
    opencv-python
    Pillow
    shapely
    pyclipper
)

cat <<EOF | docker build --network host --tag deepocr-trt -
FROM nvcr.io/nvidia/tritonserver:23.11-py3
RUN pip install --no-cache-dir --upgrade pip \
  && apt-get update && apt-get install ffmpeg libsm6 libxext6 -y \
  && pip install --no-cache-dir ${requirements[@]}
EOF