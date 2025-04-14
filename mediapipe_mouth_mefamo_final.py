
import cv2
import mediapipe as mp
import time
import numpy as np
import onnxruntime as ort

# Load MeFaMo ONNX model
model_path = "mefamo.onnx"  # make sure this file is in the same folder
session = ort.InferenceSession(model_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
prev_time = 0

MOUTH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LEFT_MOUTH_CORNER_IDX = 61
RIGHT_MOUTH_CORNER_IDX = 291
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

def run_mefamo_inference(landmarks):
    input_data = np.array(landmarks, dtype=np.float32).flatten()[np.newaxis, :]
    outputs = session.run(None, {"input": input_data})
    result = outputs[0].flatten()
    return {
        "AU1": result[0],   # Inner brow raise
        "AU4": result[3],   # Frown
        "AU6": result[5],   # Cheek raise
        "AU17": result[12]  # Jaw clench
    }

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            ih, iw, _ = image.shape

            for idx in MOUTH_LANDMARKS:
                lm = landmarks[idx]
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            left_corner = landmarks[LEFT_MOUTH_CORNER_IDX]
            right_corner = landmarks[RIGHT_MOUTH_CORNER_IDX]
            upper_lip = landmarks[UPPER_LIP_IDX]
            lower_lip = landmarks[LOWER_LIP_IDX]

            left_x = int(left_corner.x * iw)
            right_x = int(right_corner.x * iw)
            upper_y = int(upper_lip.y * ih)
            lower_y = int(lower_lip.y * ih)

            mouth_width = right_x - left_x
            mouth_height = lower_y - upper_y
            mouth_ratio = mouth_width / mouth_height if mouth_height != 0 else 0

            mouth_status = "Mouth Open" if mouth_height > 10 else "Mouth Closed"
            smile_status = "Smiling" if mouth_ratio > 2.0 else "Not Smiling"

            cv2.putText(image, mouth_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, smile_status, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Run MeFaMo inference
            points_3d = [(lm.x, lm.y, lm.z) for lm in landmarks]
            expression = run_mefamo_inference(points_3d)

            brow_status = "Raised" if expression["AU1"] > 0.5 else "Not Raised"
            cheek_status = "Puffed" if expression["AU6"] > 0.5 else "Not Puffed"
            frown_status = "Frowning" if expression["AU4"] > 0.5 else "Not Frowning"
            jaw_status = "Clenched" if expression["AU17"] > 0.5 else "Relaxed"

            cv2.putText(image, f"Brow: {brow_status}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Cheek: {cheek_status}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Frown: {frown_status}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Jaw: {jaw_status}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Facial Muscle Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
