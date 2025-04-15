# Real-Time-Facial-Muscle-Activation-Detection-Using-MediaPipe-and-MeFaMo
## Objective

This project uses **MediaPipe**, **MeFaMo**, and **PyLiveLinkFace** to detect facial Action Units (AUs) in real time using your webcam.  
It includes a **manual calibration feature** to personalize detection based on your unique facial structure.
## Background

The original script detected only mouth status (open/closed) and smiling using 2D facial landmarks. This updated implementation uses 3D facial landmarks and blendshape estimation to extract and visualize multiple facial muscle activations.
# Why calibrate?

Facial structures and resting expressions differ from person to person.  
Calibration allows the system to:

- Adapt to your **neutral face**
- Eliminate false positives from slight natural movements (like a resting smile)
- Improve the accuracy and responsiveness of AUs

## Approach

In my solution I used :

MediaPipe FaceMesh for real-time 3D face landmark detection.

MeFaMo (MediaPipe Face Mocap) for calculating blendshapes & Converts 2D landmarks to metric 3D landmarks using PCF.

BlendshapeCalculator from MeFaMo to compute Action Units (AUs) based on normalized and metric 3D landmarks.

OpenCV to overlay facial wireframe and display AU intensity values live on webcam.

| Action Unit | Description     | Blendshape Detected                  |
|-------------|------------------|--------------------------------------|
| AU1         | Brow Raise       | BrowInnerUp                          |
| AU4         | Frown            | BrowDownLeft, BrowDownRight          |
| AU6         | Cheek Puff       | CheekSquintLeft, CheekSquintRight    |
| AU12        | Smile            | MouthSmileLeft, MouthSmileRight      |
| AU17        | Jaw Clench       | JawOpen (inverted)                   |


 ## Key Functions and Classes Used
| Function / Class                          | Purpose                                |
|-------------------------------------------|----------------------------------------|
| `mp.solutions.face_mesh.FaceMesh`         | Detect facial landmarks                |
| `get_metric_landmarks()`                  | Convert normalized landmarks to 3D     |
| `PCF()`                                   | Defines camera parameters for 3D geometry |
| `BlendshapeCalculator.calculate_blendshapes()` | Computes AU activations         |
| `live_link_face.get_blendshape()`         | Retrieves smooth AU values             |
| `cv2.putText()`                           | Displays each AU and intensity         |


##  Before and After Calibration Demo

###  Before Calibration
This shows AU values responding even at rest (no baseline subtraction).
 [Watch Before Calibration](https://drive.google.com/file/d/1VYnt-JPNPpDoVEKYLit7D_Y4-sMb-10g/view?usp=sharing)

### After Calibration
This shows the effect of neutral face calibration. AU values increase only with intentional expressions.
[Watch After Calibration](https://drive.google.com/file/d/1YGC1cKLqvfTTuke4WHIwDOLQh6lUy84g/view?usp=sharing)


