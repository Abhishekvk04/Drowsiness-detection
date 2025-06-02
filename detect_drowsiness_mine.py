import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import mediapipe as mp
# import os
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Now open the RTSP stream with OpenCV using FFmpeg backend

def sound_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Constants
EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False
ALARM_SOUND_PATH = "alarm.wav"  # <-- replace with your path if needed

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for eyes (based on MediaPipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start video stream
print("[INFO] Starting webcam...")
# cap = cv2.VideoCapture("rtsp://admin:admin123@10.101.0.20:554/avstream/channel=2/stream=0.sdp", cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            def get_landmark_coords(indices):
                return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

            leftEye = get_landmark_coords(LEFT_EYE)
            rightEye = get_landmark_coords(RIGHT_EYE)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours
            for (x, y) in leftEye + rightEye:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm, args=(ALARM_SOUND_PATH,))
                        t.daemon = True
                        t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()