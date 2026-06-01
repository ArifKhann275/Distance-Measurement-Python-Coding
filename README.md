# Monocular Video Distance Tracking (YOLOv8 + MiDaS + Optical Flow)

This repository contains a Python script that estimates the real-world distance traveled by individuals in a standard 2D video. It combines object detection, monocular depth estimation, and optical flow to calculate displacement in both lateral (x, y) and depth (z) axes.

## Methodology

Measuring absolute distance from a single uncalibrated camera is an ill-posed problem. This script attempts to solve it by combining three techniques:

1. **Detection:** YOLOv8 (`yolov8n.pt`) is used to detect and draw bounding boxes around people in the frame.
2. **Depth Estimation:** MiDaS (`intel-isl/MiDaS`) generates a relative depth map of the scene. The median depth value within the center of a detected bounding box is used as the object's z-axis position.
3. **Lateral Tracking:** Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) extracts and tracks keypoints (using `goodFeaturesToTrack`) within the bounding box to measure pixel displacement between frames.
4. **Distance Calculation:** To convert pixel and relative depth changes into real-world meters, the script requires a reference scale. By default, it uses an **Assumed Height** constraint (1.70 meters). It calculates the `meters_per_pixel` ratio based on the bounding box height. 
   Total displacement per frame is calculated as the hypotenuse of the lateral step and the depth step.

## Requirements

The code is heavily dependent on PyTorch and OpenCV. A CUDA-enabled GPU is highly recommended for real-time processing.

```bash
pip install opencv-python numpy torch torchvision ultralytics
