# Lane Detection and Obstacle Detection

This project implements a vision-based lane detection pipeline for autonomous driving scenarios. The system works mainly in the Bird's Eye View (BEV) domain, where the perspective distortion of the road scene is reduced and lane markings can be processed in a simpler geometric representation.

The lane detection approach is inspired by the GOLD algorithm and is extended with photometric preprocessing, feature fusion, binary mask refinement, geometric lane fitting, temporal stabilization, and obstacle detection using YOLO.

## Project Overview

Lane detection is an important task in autonomous driving because it allows a vehicle to estimate its position with respect to the road boundaries. Real driving scenes can be difficult to process because lane markings may be affected by shadows, occlusions, weak visibility, parked vehicles, road texture, intersections, and non-flat road surfaces.

The goal of this project is to build a complete image-processing pipeline that receives a road image or video frame, transforms it into a BEV representation, extracts lane features, estimates lane boundaries, detects relevant obstacles, and visualizes the final result both in the original image and in the BEV view.

The complete processing of each frame is handled by a main pipeline function that applies all the steps from the input image to the final visualization.

## Main Features

- Bird's Eye View generation using inverse perspective mapping
- Photometric preprocessing for contrast and illumination normalization
- GOLD-based lane feature extraction
- Feature fusion with top-hat filtering and Sobel gradients
- Binary segmentation using global and adaptive thresholding
- Morphological and structural mask refinement
- Histogram-based lane initialization
- Sliding-window lane boundary fitting
- Lane validation and ego-lane selection
- Continuous/dashed lane type classification
- Temporal stabilization of lane estimates
- Clutter rejection in complex scenarios
- YOLO-based obstacle detection
- Obstacle distance and lateral-position estimation
- Visualization on both original image and BEV representation

## Camera Model and BEV Representation

The system uses a pinhole camera model to describe the projection between the road plane and the image plane. The camera parameters used by the pipeline are:

| Parameter | Value |
|---|---:|
| Image resolution | 1920 x 1080 pixels |
| Focal length | fx = fy = 1970 px |
| Principal point | cx = 970 px, cy = 483 px |
| Camera height | 1.660 m |
| Pitch angle | 0 degrees |

The projection model is:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

The BEV representation is generated under the flat-road assumption. In this project, the BEV covers the road area in front of the vehicle with:

| BEV parameter | Value |
|---|---:|
| Lateral range | [-4.8, +4.8] m |
| Forward range | [5.5, 35.0] m |
| BEV resolution | 1000 x 1200 pixels |

Working in BEV makes the lane markings appear more regular and with a more uniform width, which simplifies the following image-processing and geometric-estimation steps.

## Processing Pipeline

For each frame, the system applies the following stages:

1. BEV generation
2. Photometric preprocessing
3. Lane feature extraction
4. Binary segmentation
5. Mask refinement
6. Lane detection and fitting
7. Obstacle detection
8. Final visualization

Some components, such as the inverse perspective mapping tables and the homography, are precomputed at the beginning to reduce the computational cost during frame-by-frame processing.

## Inverse Perspective Mapping

The first step consists of transforming the original image into a BEV image. This is done using inverse perspective mapping.

The transformation maps image coordinates into ground-plane coordinates:

```text
(u, v) -> (X, Y)
```

Each BEV pixel corresponds to a point on the road plane. Since not every BEV pixel corresponds to a valid image coordinate, a valid mask is also built to remove areas outside the projected region.

This step reduces perspective effects and makes lane markings easier to analyze.

## BEV Preprocessing

After the BEV image is generated, it is converted to grayscale and enhanced before lane feature extraction. The preprocessing stage improves lane visibility and reduces the effect of non-uniform illumination.

The operations include:

- contrast stretching
- CLAHE for local contrast enhancement
- Gaussian smoothing for noise reduction
- gamma correction

These operations are useful in the presence of shadows, reflections, weak markings, and illumination changes.

## Lane Feature Extraction

Lane markings are usually brighter than the surrounding road surface. The GOLD-inspired filter exploits this property by computing a lateral response:

```text
R(x) = max(0, (I(x) - I(x - m)) + (I(x) - I(x + m)))
```

The response is high when a pixel is brighter than its left and right neighbors at a given distance. This behavior is typical of lane markings in the BEV image.

In this project, the GOLD response is combined with additional cues:

- top-hat filtering, used to highlight thin bright structures
- Sobel gradient, used to emphasize intensity changes and edges

Combining multiple cues gives a more stable feature map than using the GOLD response alone, especially when markings are weak or the road surface is not uniform.

## Binary Segmentation and Mask Refinement

The lane feature map is converted into a binary mask using a combination of:

- global thresholding
- adaptive thresholding

This produces an initial separation between possible lane pixels and background pixels.

The mask is then refined using morphological and structural filtering. The refinement stage includes:

- keeping mainly vertical structures, which are more compatible with lane markings in BEV
- removing wide connected components that are unlikely to be lanes
- restricting the search to the central ego-road corridor

This step removes many false positives caused by shadows, road texture, crosswalks, parked vehicles, or other visual patterns. The output is a cleaner binary mask that can be used for lane boundary estimation.

## Lane Detection

Starting from the refined binary mask, the goal is to estimate the lane boundaries and obtain a stable geometric representation of the ego lane.

### Histogram-Based Initialization

A column-wise histogram is computed on the binary mask:

```text
H(x) = sum_y mask(x, y)
```

This histogram counts how many active pixels are present in each column. Since lane markings appear as vertical structures in BEV, they tend to generate peaks in the histogram. These peaks are used to select initial seed positions for the left and right lane boundaries.

### Sliding-Window Boundary Fitting

Starting from the initial seeds, a sliding-window search is performed from the bottom to the top of the BEV image. At each step, the algorithm searches for lane pixels and updates the window position.

The collected points are then approximated with a linear model:

```text
x = a*y + b
```

Outliers are removed using the residual error, which improves robustness when the binary mask still contains noise.

### Boundary Validation and Ego-Lane Selection

Not all detected lines correspond to valid lane boundaries. Candidate boundaries are checked using geometric and statistical criteria, including constraints on lane width and relative position.

The final ego lane is selected by choosing the best combination of left and right boundaries.

### Clutter Rejection

In complex scenes, such as intersections or areas with many road markings, the binary mask may contain a large amount of noise. To avoid unstable results, the system evaluates indicators such as:

- horizontal energy ratio
- center occupancy ratio

If the scene is considered too cluttered, lane detection is skipped for that frame. This prevents the system from returning unreliable lane estimates in situations where the lane structure is not clearly defined.

### Lane Type Classification

The system also classifies lane boundaries as continuous or dashed. The classification is based on the analysis of a vertical strip around each detected boundary.

The criteria include:

- number of active pixels along the column
- number of separated segments
- length of the longest continuous segment

The classification is smoothed over time to avoid unstable frame-by-frame changes.

### Temporal Stabilization

The detected lane positions are filtered over time using an exponential moving average:

```text
x_smooth(t) = alpha * x_smooth(t-1) + (1 - alpha) * x(t)
```

This reduces noise and prevents sudden changes between consecutive frames. Unrealistic jumps are ignored, and the system keeps track of lane position, lane width, and lane center over time. If no lanes are detected for several consecutive frames, the internal state is reset.

## Obstacle Detection

The project also includes an obstacle detection module based on YOLO. The detector is applied directly to the original image and returns bounding boxes with confidence scores and class labels.

Only classes relevant for driving are considered, such as:

- car
- person
- bus
- truck

For each detected object, the bottom point of the bounding box is used to estimate its distance from the vehicle under the flat-road assumption:

```text
Z = fy * h / (v - v_horizon)
```

The lateral position is also estimated so that the object can be represented in the BEV domain.

Obstacle relevance depends on the lane detection result:

- if both lane boundaries are available, only obstacles inside the ego lane are considered
- if one or both lane boundaries are missing, all detected obstacles are shown

## Visualization

The final output consists of two synchronized views:

- original image
- BEV representation

In the BEV view, the lane boundaries are drawn as vertical markers. Continuous lanes are displayed as solid lines, while dashed lanes are shown as separated segments. If no valid lane is detected, the system displays a "No lanes found" message.

In the original image, the detected lanes are projected back from the BEV and drawn on the road scene. Obstacles are shown with bounding boxes, labels, and estimated distance.

An optional mask visualization can also be enabled to inspect the intermediate binary mask used for lane detection.

## Results

The system was tested on several driving sequences under different conditions.

### Changing Road Geometry

In road segments with clear lane markings and mostly flat geometry, the system produces lane estimates that are well aligned both in the original image and in the BEV view. When the road becomes sloped, the flat-road assumption introduces distortions in the BEV. However, the lane structure is still detected, and the result becomes more accurate again when the road returns to a flat configuration.

### Intersections and Undefined Lane Structure

In intersection scenarios, the system can correctly avoid detecting lanes and display the "No lanes found" message. This behavior is important because it prevents incorrect lane estimations when the lane structure is not clearly defined.

Some false detections may still appear when parts of vehicles or road structures look similar to lane markings, but these effects are usually limited.

### Urban Traffic and Parked Vehicles

In urban scenarios with moving traffic and parked vehicles, the system remains generally stable and is able to detect both lanes and obstacles. Parked vehicles and complex road markings can introduce noise, especially on the right side of the image, and may slightly affect the estimated lane position.

## Limitations

The main limitations of the system are:

- dependence on the flat-road assumption
- sensitivity to strong slopes or non-planar road geometry
- possible confusion between lane markings and similar visual patterns
- noise introduced by parked vehicles, intersections, road texture, or complex markings
- occasional false lane detections in cluttered scenes

Despite these limitations, the pipeline provides a stable and reliable solution in most standard road scenarios.

## Technologies

The project is implemented in Python and uses computer vision and deep learning tools for lane and obstacle detection.

Main technologies include:

- Python
- OpenCV
- NumPy
- YOLO
- image processing in Bird's Eye View
- inverse perspective mapping
- morphological filtering
- sliding-window lane fitting

## Conclusion

This project implements a lane detection system based on a BEV representation and a GOLD-inspired feature extraction approach. The pipeline combines photometric preprocessing, feature fusion, mask refinement, geometric fitting, and temporal stabilization to obtain robust lane estimates.

The YOLO obstacle detection module extends the system by identifying relevant objects in the driving scene and estimating their position with respect to the ego lane.

Overall, the system performs well in standard road conditions and remains reasonably stable in more complex urban scenarios. Its main weaknesses are related to the flat-road assumption and to visual patterns that can be confused with lane markings.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
