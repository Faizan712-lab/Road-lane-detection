# 🚗 Road Lane Detection using OpenCV

This project detects lane lines on roads using classical Computer Vision techniques like **Canny Edge Detection** and **Hough Line Transform** — no deep learning required.

## 🧠 Overview
This program identifies left and right lane boundaries in road videos and overlays them with colored lines and a green-filled lane area.

## ⚙️ Tech Stack
- **Python 3.x**
- **OpenCV** for computer vision
- **NumPy** for mathematical operations

## 🧩 Algorithms Used
1. **Grayscale Conversion** – simplifies image for edge detection  
2. **Gaussian Blur** – reduces noise  
3. **Canny Edge Detection** – identifies strong gradients (edges)  
4. **Region of Interest Masking** – focuses only on the road area  
5. **Hough Line Transform** – detects lane line segments  
6. **Slope Averaging** – smooths multiple detections into clean lines  
7. **Overlay Drawing** – displays red (left), blue (right), and green (lane fill)

## 🎥 How to Run
```bash
git clone https://github.com/<your-username>/Road-Lane-Detection.git
cd Road-Lane-Detection
pip install -r requirements.txt
python LaneDetection.py
Press 'q' to quit the video window.
