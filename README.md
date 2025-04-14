# Real-Time-Facial-Muscle-Activation-Detection-Using-MediaPipe-and-MeFaMo
## Objective

This project extends the mediapipe_mouth_open_close.py script to detect real-time facial muscle activations (action units) using 3D face mesh data from MediaPipe and the MeFaMo library.

## Background

The original script detected only mouth status (open/closed) and smiling using 2D facial landmarks. This updated implementation uses 3D facial landmarks and blendshape estimation to extract and visualize multiple facial muscle activations.

## Approach

In my solution i used :

MediaPipe FaceMesh for real-time 3D face landmark detection.

...MeFaMo (MediaPipe Face Mocap) for calculating blendshapes & Converts 2D landmarks to metric 3D landmarks using PCF.

BlendshapeCalculator from MeFaMo to compute Action Units (AUs) based on normalized and metric 3D landmarks.

OpenCV to overlay facial wireframe and display AU intensity values live on webcam.

 ---------------------------------------------------------------------- 
 Action Unit      ,Description                ,Blendshape detected              
-------------------------------------------------------------------------
  AU1            | ,Brow raise                  |, BrowInnerUp
  Au4            | ,Frown                       |,BrowDownLeft,BrowDownRight
  Au6            | ,cheekpuff                   |,cheekSquintLeft/Right
  Au12           | ,Smile                       |,MouthSmileleft/right
  Au17           | ,Jaw Clench                  |,jaw open


                          Key Functions and Classes Used----------------------------
 -------------------------------------------------------------------------------
 Function / Class                       |        Purpose 
 -------------------------------------------------------------------------------
 mp.solutions.face_mesh.FaceMesh        |         Detect facial landmarks
 get_metric_landmarks()                 |         Convert normalized landmarks to 3D
 PCF()                                  |         Defines camera parameters for 3D geometry
 BlendshapeCalculator.calculate_blendshapes() |   Computes AU activations
 live_link_face.get_blendshape()        |         Retrieves smooth AU values
 cv2.putText()                          |         Displays each AU and intensity
