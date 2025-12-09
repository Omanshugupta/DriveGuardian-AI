import cv2
import numpy as np
import os
import pickle
from datetime import datetime

class SimpleFaceRecognition:
    def __init__(self, database):
        self.database = database
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_size = (200, 200)  # width, height for training/prediction
        self.load_recognizer()

    def _prepare_image(self, img):
        if img is None: return None
        # Grayscale and histogram equalize before resize
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        try:
            img = cv2.resize(img, self.face_size)
        except Exception:
            return None
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img

    def load_recognizer(self):
        self.known_faces = {}
        drivers = self.database.get_all_drivers()
        if len(drivers) == 0:
            self.recognizer_labels = {}
            print("No drivers in database yet.")
            return
        try:
            images = []
            labels = []
            label_id = 0
            for driver in drivers:
                try:
                    face_data = pickle.loads(driver[4])
                    if isinstance(face_data, list):
                        for face_img in face_data:
                            prepared = self._prepare_image(face_img)
                            if prepared is not None:
                                images.append(prepared)
                                labels.append(label_id)
                    else:
                        prepared = self._prepare_image(face_data)
                        if prepared is not None:
                            images.append(prepared)
                            labels.append(label_id)
                    self.known_faces[label_id] = driver[1]
                    label_id += 1
                except Exception as e:
                    print(f"Error loading driver data: {e}")
                    continue
            if len(images) > 0 and len(labels) > 0:
                self.recognizer.train(images, np.array(labels))
                self.recognizer_labels = self.known_faces
                print(f"Loaded {len(set(labels))} driver(s) with {len(images)} face samples.")
            else:
                print("No valid face data found in database.")
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            self.recognizer_labels = {}

    def capture_face_for_training(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        print("Position your face in front of the camera. Press SPACE to capture.")
        face_samples = []
        while len(face_samples) < 20:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Capture Face Samples', frame)
            key = cv2.waitKey(1) & 0xFF
            # Only allow the sample if exactly 1 face
            if key == ord(' '):
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = self._prepare_image(face_roi)
                    if face_roi is not None:
                        face_samples.append(face_roi)
                    print(f"Captured sample {len(face_samples)}/20")
                elif len(faces) == 0:
                    print("No face detected for sample. Try again.")
                else:
                    print("More than one face detected. Only single face allowed for sample.")
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if len(face_samples) >= 5:
            return face_samples
        return None

    def recognize_face_in_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            print("[Debug] No face detected for recognition.")
        elif len(faces) > 1:
            print("[Debug] More than one face detected for recognition.")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for (x, y, w, h) in faces:
            if len(self.known_faces) == 0:
                return None
            face_roi = gray[y:y+h, x:x+w]
            face_roi = self._prepare_image(face_roi)
            if face_roi is None:
                continue
            try:
                label, confidence = self.recognizer.predict(face_roi)
                if confidence < 100:
                    name = self.known_faces.get(label, "Unknown")
                    return name
            except Exception as e:
                print(f"[Debug] recognizer error: {e}")
                return None
        return None

    def register_new_driver(self, name, age, license_no):
        print(f"\n--- Registering New Driver: {name} ---")
        print("Keep your face still. The system will automatically capture samples.")
        face_samples = self.capture_face_for_training()
        if face_samples is None or len(face_samples) < 5:
            print("Failed to capture enough face samples.")
            return False
        import pickle
        encoded_face = pickle.dumps(face_samples)
        driver_id = self.database.add_driver(name, age, license_no, encoded_face)
        if driver_id:
            self.load_recognizer()
            print(f"Driver {name} registered successfully!")
            return True
        else:
            print(f"Failed to register driver. License number might already exist.")
            return False
    def get_face_from_camera(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            return face_roi
        return None
