import cv2
import numpy as np
from scipy.spatial import distance as dist

class DrowsinessDetector:
    def __init__(self):
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
        self.EAR_CONSEC_FRAMES = 30  # Number of consecutive frames for drowsiness (about 1 second at 30 FPS)
        self.CLOSED_DURATION_SECONDS = 3  # 3 seconds for alert
        self.FRAME_RATE = 30  # Assuming 30 FPS

        self.eye_closed_frames = 0
        self.blink_counter = 0
        self.total_closed_time = 0.0
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # Landmarks for face (simplified approach)
        self.landmark_model = None

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_drowsiness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        drowsiness_detected = False
        debug_found_face = False

        for (x, y, w, h) in faces:
            debug_found_face = True
            roi_gray = gray[y:y+h, x:x+w]
            # Only use upper half of face ROI for eye detection
            eye_roi_area = roi_gray[0:int(h/2), :]
            # Histogram normalize for consistency
            eye_roi_area = cv2.equalizeHist(eye_roi_area)
            roi_color = frame[y:y+h, x:x+w]
            # Eye detection parameters can be adjusted if unstable
            eyes = self.eye_cascade.detectMultiScale(eye_roi_area, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                # Draw debug rectangles for detected eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            if len(eyes) == 0:
                print("[Debug] No eyes detected in upper face ROI. Assuming closed.")
                self.eye_closed_frames += 1
                drowsiness_detected = True
            elif len(eyes) < 2:
                print("[Debug] Only one eye detected. Might be closed or occluded.")
                self.eye_closed_frames += 1
                drowsiness_detected = True
            else:
                ear_values = []
                for (ex, ey, ew, eh) in eyes:
                    single_eye = eye_roi_area[ey:ey+eh, ex:ex+ew]
                    brightness = np.mean(single_eye)
                    if brightness < 30:
                        ear_values.append(0.15)
                    else:
                        ear_values.append(0.25)
                if len(ear_values) > 0:
                    ear = np.mean(ear_values)
                    if ear < self.EAR_THRESHOLD:
                        self.eye_closed_frames += 1
                    else:
                        if self.eye_closed_frames >= self.EAR_CONSEC_FRAMES:
                            drowsiness_detected = True
                        self.eye_closed_frames = 0
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if not debug_found_face:
            print("[Debug] No face detected in frame.")
        closed_time = self.eye_closed_frames / self.FRAME_RATE
        if closed_time >= self.CLOSED_DURATION_SECONDS:
            return True, closed_time, frame
        return drowsiness_detected, closed_time, frame
    def reset(self):
        self.eye_closed_frames = 0
        self.total_closed_time = 0.0
