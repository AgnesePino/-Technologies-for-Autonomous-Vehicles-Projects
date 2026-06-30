# Autonomous Driving Projects

This repository collects three projects developed for the Autonomous Driving course.  
Each project focuses on a different component of an autonomous or intelligent vehicle system: global path planning, lane and obstacle detection, and driver monitoring.

## Projects Overview

### 1. Global Path Planning: Dijkstra vs A*

The first project compares two classical graph-search algorithms for global path planning: **Dijkstra** and **A\***.

The road networks are extracted from OpenStreetMap using OSMnx and represented as directed graphs, where nodes correspond to intersections and edges correspond to road segments. The edge cost is based on estimated travel time, computed from road length and speed.

The experiments are performed on two cities with different graph sizes:

- **Turin**, a large and dense urban road network
- **Aosta**, a smaller road network

For A*, three heuristics are tested:

- Manhattan distance
- Euclidean distance
- Haversine distance

The results show that A* reduces the number of expanded nodes compared to Dijkstra, while Euclidean and Haversine heuristics preserve the optimal path. Manhattan is often more aggressive and faster, but in some cases it can produce a slightly suboptimal path.

---

### 2. Lane Detection and Obstacle Detection

The second project implements a lane detection pipeline inspired by the **GOLD algorithm**.

The system transforms the original camera image into a **Bird’s Eye View** using inverse perspective mapping. In the BEV domain, lane markings are easier to analyze because the perspective distortion is reduced.

The pipeline includes:

- BEV generation
- photometric preprocessing
- GOLD-based lane feature extraction
- top-hat and Sobel feature fusion
- binary segmentation
- morphological mask refinement
- histogram-based lane initialization
- sliding-window lane boundary fitting
- lane type classification
- temporal stabilization

The system also includes an obstacle detection module based on **YOLO**. Detected objects are filtered according to their relevance to the ego lane and visualized both in the original image and in the BEV representation.

The project was tested on multiple driving sequences, including flat roads, sloped roads, intersections, urban traffic, parked vehicles, and complex road markings. The main limitations are related to the flat-road assumption and visual patterns that may be confused with lane markings.

---

### 3. Driver Monitoring System

The third project implements a vision-based **Driver Monitoring System** using a standard laptop camera.

The system processes the driver’s face in real time using MediaPipe Face Landmarker. It estimates head pose, iris position, eye aperture, and facial regions used for heart-rate extraction.

The system detects:

- **Owl distraction**, caused mainly by head rotation
- **Lizard distraction**, caused mainly by eye movement while the head remains frontal
- **Microsleep**, based on prolonged eye closure
- **Sleep**, based on longer eye closure
- **Heart rate**, estimated using remote photoplethysmography

An initial calibration phase is used to compute a personal baseline for head position, iris position, and eye openness. This makes the system more robust to differences between users, webcam position, and lighting conditions.

The final output is an annotated video stream showing the driver state and estimated heart rate. The processed video is also saved as an output file.

---

## Repository Structure

```text
autonomous-driving-projects/
│
├── project-1-path-planning/
│   ├── src/
│   ├── report/
│   └── README.md
│
├── project-2-lane-detection/
│   ├── src/
│   ├── report/
│   └── README.md
│
├── project-3-driver-monitoring/
│   ├── src/
│   ├── report/
│   └── README.md
│
└── README.md
```
## Topics Covered
Autonomous driving
Global path planning
Dijkstra algorithm
A* algorithm
OpenStreetMap road networks
Lane detection
Bird’s Eye View
Inverse Perspective Mapping
GOLD algorithm
YOLO obstacle detection
Driver Monitoring System
MediaPipe Face Landmarker
Distraction detection
Microsleep and sleep detection
rPPG heart-rate estimation

## Technologies
The projects are mainly implemented in Python and use different computer vision, graph-processing, and machine-learning tools, including:
Python
OpenCV
NumPy
OSMnx
NetworkX
Matplotlib
MediaPipe
YOLO
Goal of the Repository

The goal of this repository is to collect and document three independent projects related to autonomous driving systems. Together, they cover different levels of vehicle autonomy:

Planning, by computing efficient routes on real road networks
Perception, by detecting lanes and obstacles from camera images
Human monitoring, by analyzing driver distraction and fatigue

These projects show how graph algorithms, image processing, machine learning, and real-time computer vision can be combined to support intelligent transportation systems.

## License

This project is licensed under the MIT License.