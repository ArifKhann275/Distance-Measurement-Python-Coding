# 🎯 Real-Time Human Distance Measurement using YOLOv8, MiDaS and Optical Flow

## 📌 Project Overview

This project estimates the distance traveled by people in a video using computer vision techniques. The system combines object detection, depth estimation, and optical flow tracking to measure human movement in real-world units (meters).

The main goal of this project was to explore how multiple computer vision models can be integrated to perform approximate real-world distance measurement from a single video stream.

---

## ✨ Features

* Person detection using YOLOv8
* Monocular depth estimation using MiDaS
* Optical Flow-based motion tracking
* Multi-person tracking with unique IDs
* Real-time distance estimation
* Video output generation with annotations
* Optional manual calibration using known real-world distances

---

## ⚙️ How It Works

### 1. Person Detection

YOLOv8 detects people in each frame.

### 2. Depth Estimation

MiDaS generates a depth map for the scene.

### 3. Motion Tracking

Optical Flow tracks feature points between consecutive frames.

### 4. Distance Calculation

Lateral movement and depth changes are combined to estimate displacement.

### 5. Distance Accumulation

The total traveled distance is accumulated for each tracked person.

---

## 🛠️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* MiDaS
* PyTorch
* NumPy

---

## 📏 Distance Estimation Strategy

Since a monocular camera does not provide true depth information, distance estimation is approximated using:

* Human height assumption (default: 1.7 m)
* MiDaS relative depth values
* Optical Flow displacement
* Optional user calibration

The estimated distance should therefore be considered an approximation rather than a precise measurement.

---

## 📊 Results

The system can:

* Detect and track multiple people
* Visualize depth information
* Estimate traveled distance in meters
* Generate annotated output videos

---

## 🚀 Future Improvements

* Integrate DeepSORT or ByteTrack for more robust tracking
* Improve distance calibration methods
* Support real-time webcam deployment
* Explore stereo vision for more accurate depth estimation
* Evaluate performance on crowded scenes

---

## 🖼️ Output Examples

### Person Detection and Tracking

(Add screenshot here)

### Depth Map Visualization

(Add screenshot here)

### Distance Estimation Output

(Add screenshot here)

---

## 📝 Notes

This project was developed as a learning and research-oriented computer vision experiment. The focus was on combining detection, depth estimation, and tracking techniques to estimate human movement from monocular video data.
