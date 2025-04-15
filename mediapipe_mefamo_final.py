import cv2
import mediapipe as mp
import numpy as np
import time
from mefamo.blendshapes.blendshape_calculator import BlendshapeCalculator
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from mefamo.custom.face_geometry import PCF, get_metric_landmarks

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera resolution
image_width, image_height = 640, 480

# MeFaMo setup
blendshape_calculator = BlendshapeCalculator()
live_link_face = PyLiveLinkFace(fps=30, filter_size=4)

# PCF setup
camera_matrix = np.array(
    [[image_width, 0, image_width / 2],
     [0, image_width, image_height / 2],
     [0, 0, 1]], dtype="double"
)
pcf = PCF(
    near=1,
    far=10000,
    frame_height=image_height,
    frame_width=image_width,
    fy=camera_matrix[1, 1],
)

# Blendshapes to track
target_shapes = [
    FaceBlendShape.MouthSmileLeft,
    FaceBlendShape.MouthSmileRight,
    FaceBlendShape.JawOpen,
    FaceBlendShape.BrowInnerUp,
    FaceBlendShape.CheekSquintLeft,
    FaceBlendShape.CheekSquintRight,
    FaceBlendShape.BrowDownLeft,
    FaceBlendShape.BrowDownRight,
]

# Calibration setup
calibrating = False
baseline_blendshapes = {}
calibration_start = None
calibration_duration = 5  # seconds
calibration_buffer = []

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark positions
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]).T
            metric_landmarks, pose_transform = get_metric_landmarks(landmarks.copy(), pcf)

            # Compute blendshapes
            blendshape_calculator.calculate_blendshapes(
                live_link_face, metric_landmarks[0:3].T, face_landmarks.landmark
            )

            # Display values
            y_offset = 30
            current_blendshapes = {}
            for shape in target_shapes:
                value = live_link_face.get_blendshape(shape)
                current_blendshapes[shape.name] = value

                if calibrating:
                    calibration_buffer.append(current_blendshapes)
                    cv2.putText(image, "Calibrating... Hold neutral face", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(image, f"{shape.name}: {value:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
                else:
                    baseline = baseline_blendshapes.get(shape.name, 0.0)
                    calibrated_value = max(0.0, value - baseline)
                    cv2.putText(image, f"{shape.name}: {calibrated_value:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

    # End calibration after duration
    if calibrating and (time.time() - calibration_start > calibration_duration):
        if len(calibration_buffer) > 0:
            print("Calibration complete!")
            averaged_blendshapes = {}
            for shape in target_shapes:
                values = [frame[shape.name] for frame in calibration_buffer if shape.name in frame]
                averaged_blendshapes[shape.name] = np.mean(values) if values else 0.0
            baseline_blendshapes = averaged_blendshapes
        else:
            print("Calibration failed: no valid frames.")
        calibrating = False
        calibration_buffer.clear()

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (10, image_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display image
    cv2.imshow('MeFaMo AU Detection (Press C to calibrate)', image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c') and not calibrating:
        print("Starting manual calibration...")
        calibrating = True
        calibration_start = time.time()
        calibration_buffer = []

cap.release()
cv2.destroyAllWindows()
