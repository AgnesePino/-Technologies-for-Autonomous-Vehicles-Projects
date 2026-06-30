# Driver Monitoring System

This project implements a real-time **Driver Monitoring System (DMS)** using a standard laptop camera. The system analyzes the driver's face frame by frame to detect distraction, fatigue-related events, and estimate heart rate using remote photoplethysmography.

The pipeline combines facial landmark detection, gaze analysis, eye-closure monitoring, temporal logic, and rPPG-based heart-rate estimation. The final output is an annotated video stream that displays the current driver state and the estimated BPM.

## Project Overview

Driver Monitoring Systems are designed to improve road safety by observing the driver's state and identifying potentially unsafe behaviors such as distraction, fatigue, drowsiness, or prolonged eye closure.

This project focuses on three main aspects:

- **Distraction detection**, based on head and eye movement
- **Fatigue detection**, based on prolonged eye closure
- **Heart-rate estimation**, based on subtle skin-color variations captured by the camera

The system is implemented in Python and processes video frames in real time. Each frame is acquired from the laptop camera, mirrored, converted from BGR to RGB, and processed using MediaPipe Face Landmarker. The detected landmarks are then used to estimate head orientation, iris position, eye aperture, and facial regions useful for rPPG.

## Main Features

The implemented system can detect and display the following states:

- **Focused on the road**
- **Distracted (long)**
- **Distracted (short)**
- **Microsleep**
- **Sleep**

Internally, distraction is divided into two categories:

- **Owl distraction**: distraction mainly caused by head rotation
- **Lizard distraction**: distraction mainly caused by eye movement while the head remains frontal

The system also estimates the driver's heart rate and shows the BPM value on the video output. This value is an approximate estimate and should not be considered a clinical measurement.

## Processing Pipeline

The complete pipeline is applied independently to each frame and includes:

1. Camera acquisition and frame preprocessing
2. Face landmark detection using MediaPipe Face Landmarker
3. Initial calibration of head pose, iris position, and eye aperture
4. Head-based gaze estimation for owl distraction
5. Iris-based gaze estimation for lizard distraction
6. Eye-closure detection for blink filtering, microsleep, and sleep
7. Heart-rate estimation using rPPG
8. Final state selection and visualization
9. Annotated video saving

The main script processes the video stream in real time and saves the annotated output as:

```text
DMS_output.avi
```

During execution, the system can be stopped by pressing `q` or `Esc`. A debug mode can also be enabled to visualize selected landmarks and internal timers.

## Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- NumPy
- FastICA
- FFT-based signal analysis

## Face Landmark Detection and Calibration

The monitoring pipeline starts with facial landmark detection. MediaPipe Face Landmarker processes one face at a time and returns normalized landmark coordinates. When pixel coordinates are needed, normalized values are converted using the frame width and height:

```text
x_px = x_norm * W
y_px = y_norm * H
```

The landmarks are used to estimate:

- head position
- iris position
- eye aperture
- facial regions for heart-rate estimation

Before real-time monitoring starts, the system performs an initial calibration. During this phase, the user looks straight at the camera with open eyes and a stable head position. The system collects 60 valid samples and computes reference values for:

- frontal head position
- frontal iris position
- open-eye aperture

Calibration is necessary because face position, eye geometry, camera placement, and distance from the webcam can vary between users and setups.

The calibration uses robust statistics. The median is used as the central value, while signal variability is estimated with the median absolute deviation:

```text
sigma_robust = 1.4826 * median(|x_i - median(x)|)
```

This makes the baseline less sensitive to small movements or noisy frames. If the signal is unstable during calibration, the collected data is discarded and calibration must be repeated.

## Owl Distraction Detection

Owl distraction represents a gaze deviation mainly caused by head rotation.

The system computes a normalized horizontal head-rotation score using three geometric cues:

- nose position with respect to the midpoint of the eyes
- nose position with respect to the center of the face
- asymmetry between the two sides of the face

The score is computed as:

```text
G = 0.45 * ((x_nose - x_eye_mid) / w_face)
  + 0.35 * ((x_nose - x_face_center) / w_face)
  + 0.20 * ((d_right - d_left) / w_face)
```

where:

- `x_nose` is the horizontal position of the nose tip
- `x_eye_mid` is the midpoint between the two eyes
- `x_face_center` is the horizontal center of the face
- `w_face` is the face width
- `d_right` and `d_left` are the distances between the nose and the right/left face borders

The value is normalized by face width so that the score is less dependent on the distance from the camera. An exponential moving average is applied to reduce sudden changes caused by landmark noise:

```text
G_ema(t) = (1 - alpha_g) * G_ema(t-1) + alpha_g * G(t)
```

with:

```text
alpha_g = 0.35
```

The current head-rotation score is compared with the calibrated frontal baseline. Owl distraction is activated when the deviation exceeds the activation threshold and remains active until it falls below a lower deactivation threshold. This hysteresis avoids rapid state changes when the signal is close to the threshold.

A **long owl distraction** is detected when the head remains rotated for at least:

```text
T_owl >= 5 s
```

A **short owl distraction** is detected when repeated head rotations accumulate at least 10 seconds of distraction within a 30-second window:

```text
sum(T_owl_short) >= 10 s within 30 s
```

If the face is not detected for more than 0.5 seconds after calibration, the system treats the loss of landmarks as a possible owl distraction rather than immediately considering the driver focused.

## Lizard Distraction Detection

Lizard distraction represents a gaze deviation mainly caused by eye movement while the head remains approximately frontal.

For each eye, the center of the iris is compared with the center of the corresponding eye and normalized by the eye width. The final iris offset is the average of the two normalized offsets:

```text
I = 1/2 * ((x_iris_L - x_eye_L) / w_eye_L
        + (x_iris_R - x_eye_R) / w_eye_R)
```

The iris signal is filtered with an exponential moving average:

```text
I_ema(t) = (1 - alpha_i) * I_ema(t-1) + alpha_i * I(t)
```

with:

```text
alpha_i = 0.30
```

The filtered iris position is compared with the calibrated frontal iris baseline. Lizard distraction is detected only when:

- owl distraction is not active
- the head remains sufficiently frontal
- the iris displacement exceeds the activation threshold

This means that head-based distraction has priority over eye-based distraction. If the head is rotated, the event is classified as owl distraction, even if the iris position also changes.

A **long lizard distraction** is detected when the eyes remain away from the frontal direction for at least:

```text
T_lizard >= 5 s
```

A **short lizard distraction** is detected when the cumulative eye-based distraction time reaches 10 seconds within a 30-second window:

```text
sum(T_lizard_short) >= 10 s within 30 s
```

The short distraction warning remains active until the gaze has been frontal continuously for at least 2 seconds.

## Microsleep and Sleep Detection

Fatigue detection is based on the Eye Aspect Ratio (EAR). For each eye, the vertical distance between the upper and lower eyelid landmarks is divided by the horizontal distance between the two eye corners:

```text
EAR = ||p_top - p_bottom|| / ||p_left - p_right||
```

The final EAR value is the average between the left and right eyes:

```text
EAR_avg = (EAR_L + EAR_R) / 2
```

A lower EAR value indicates a smaller eye aperture. To make the detection more stable, the EAR signal is filtered with an exponential moving average. Two different coefficients are used:

- `alpha_close = 0.80`, to follow eye closure quickly
- `alpha_open = 0.35`, to make reopening more gradual and less sensitive to noise

The system uses two thresholds:

- a closing threshold
- an opening threshold

This introduces hysteresis: eyes are considered closed only when the EAR falls below the closing threshold, and they are considered open again only when the EAR exceeds the opening threshold.

Short blinks are filtered using a tolerance time:

```text
T_blink = 0.40 s
```

If the eyes remain closed for less than this duration, the event is considered a normal blink and is not counted as microsleep or sleep.

A **microsleep** event is detected when both eyes remain closed for at least:

```text
T_closed >= 4 s
```

A **sleep** event is detected when both eyes remain closed for at least:

```text
T_closed >= 7 s
```

The fatigue alarm remains active until the eyes have remained open continuously for at least:

```text
T_eyes_open >= 2 s
```

This prevents a very short eye reopening from immediately clearing the warning.

## Heart-Rate Estimation with rPPG

The system estimates heart rate using remote photoplethysmography. rPPG is based on very small skin-color variations caused by blood volume changes during the cardiac cycle.

Two regions of interest are selected in the lower part of the face, approximately on the cheeks. For each frame, the average color of these regions is computed and stored with a timestamp.

The rPPG buffer duration is:

```text
T_rPPG = 10 s
```

When enough samples are available, the RGB signals are:

1. resampled at 30 Hz
2. normalized
3. organized into an RGB signal matrix
4. processed using FastICA
5. analyzed with FFT

The frequency axis is converted into beats per minute:

```text
BPM = f * 60
```

Only plausible heart-rate values are considered:

```text
45 <= BPM <= 180
```

The strongest reliable frequency peak is selected. To avoid unrealistic jumps, a new BPM estimate is rejected if it differs from the previous value by more than 20 BPM. Otherwise, the displayed value is smoothed:

```text
BPM_new = 0.85 * BPM_old + 0.15 * BPM_estimated
```

Because rPPG is sensitive to lighting, shadows, motion, and camera quality, the BPM value is only an approximate estimate.

## State Logic

The final driver state is selected using a priority-based logic. Fatigue states have the highest priority because they are the most critical.

The priority order is:

1. **Sleep**, if both eyes remain closed for at least 7 seconds
2. **Microsleep**, if both eyes remain closed for at least 4 seconds
3. **Distracted (long)**, if owl or lizard distraction lasts at least 5 seconds
4. **Distracted (short)**, if cumulative owl or lizard distraction reaches 10 seconds within 30 seconds
5. **Focused on the road**, if none of the previous conditions is active

The distinction between owl and lizard is used internally to understand the type of distraction, while the final displayed label follows the general state categories.

When the driver is focused, eyes are open, and no fatigue alarm is active, the calibrated baselines are slowly adapted:

```text
B(t) = 0.995 * B(t-1) + 0.005 * x(t)
```

This allows the system to follow small natural posture changes without modifying the baseline too quickly.

## Output

The final output is an annotated video stream showing:

- selected face landmarks
- driver state
- distraction/fatigue warnings
- estimated heart rate
- optional debug timers

The processed video is saved as:

```text
DMS_output.avi
```

Possible displayed states are:

```text
Focused on the road
Distracted (long)
Distracted (short)
Microsleep
Sleep
```

## Results

The system was tested using a laptop camera under different conditions. After the initial calibration, it was able to recognize:

- focused driving
- owl distraction from head movement
- lizard distraction from iris movement
- microsleep after about 4 seconds of closed eyes
- sleep after about 7 seconds of closed eyes

Short blinks were ignored, and fatigue alarms were reset only after the eyes remained open for at least 2 seconds.

The rPPG module produced plausible BPM values under stable lighting and limited movement. However, because it is sensitive to illumination changes, shadows, head movement, and camera quality, the estimated heart rate should be interpreted as approximate.

## Limitations

The main limitations of the system are related to:

- webcam quality
- lighting conditions
- shadows
- face visibility
- head movement
- landmark detection errors
- motion artifacts affecting rPPG

The system is intended as a real-time computer vision prototype for driver monitoring and should not be used as a medical or safety-critical device without further validation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
