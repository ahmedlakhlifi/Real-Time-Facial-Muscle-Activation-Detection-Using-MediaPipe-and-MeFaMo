import cv2
import mediapipe as mp
import numpy as np
import time
from mefamo.blendshapes.blendshape_calculator import BlendshapeCalculator
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from mefamo.custom.face_geometry import PCF, get_metric_landmarks
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_face_mesh_tesselation_style, get_default_face_mesh_contours_style

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

# Initialize MeFaMo components
blendshape_calculator = BlendshapeCalculator()
live_link_face = PyLiveLinkFace(fps=30, filter_size=4)

# Setup PCF for 3D landmark conversion
camera_matrix = np.array(
    [[image_width, 0, image_width / 2], [0, image_width, image_height / 2], [0, 0, 1]], dtype="double"
)
pcf = PCF(
    near=1,
    far=10000,
    frame_height=image_height,
    frame_width=image_width,
    fy=camera_matrix[1, 1],
)

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
            # here i Extraced landmark data
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark[:468]]).T

            metric_landmarks, pose_transform = get_metric_landmarks(landmarks.copy(), pcf)

            # Compute blendshapes
            blendshape_calculator.calculate_blendshapes(
                live_link_face, metric_landmarks[0:3].T, face_landmarks.landmark
            )

            # Display AU values
            y_offset = 30
            for shape in [
                FaceBlendShape.MouthSmileLeft,
                FaceBlendShape.MouthSmileRight,
                FaceBlendShape.JawOpen,
                FaceBlendShape.BrowInnerUp,
                FaceBlendShape.CheekSquintLeft,
                FaceBlendShape.CheekSquintRight
            ]:
                value = live_link_face.get_blendshape(shape)
                cv2.putText(image, f"{shape.name}: {value:.2f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

    #  # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MeFaMo Face AU Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()