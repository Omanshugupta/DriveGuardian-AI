import cv2
import time
from database import DriverDatabase
from face_recognition_simple import SimpleFaceRecognition
from drowsiness_detection import DrowsinessDetector
from email_notifier import EmailNotifier
import os
from datetime import datetime

class DriverDrowsinessSystem:
    def __init__(self):
        self.database = DriverDatabase()
        self.face_recognition = SimpleFaceRecognition(self.database)
        self.drowsiness_detector = DrowsinessDetector()
        self.email_notifier = EmailNotifier()
        self.current_driver = None
        self.driving_active = False
        self.alert_played = False
    def verify_driver(self):
        print("\n=== Driver Verification ===")
        print("Position your face in front of the camera...")
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        verification_successful = False
        driver_name = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                recognized_name = self.face_recognition.recognize_face_in_frame(frame)
                if recognized_name and recognized_name != "Unknown":
                    driver_name = recognized_name
                    verification_successful = True
                    print(f"\n✓ Driver verified: {driver_name}")
                    cv2.putText(frame, f"Verified: {driver_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    print("\n✗ Unknown driver. Registration required.")
                    cv2.putText(frame, "Unknown Driver", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break
                cv2.putText(frame, "Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Driver Verification', frame)
                if verification_successful:
                    time.sleep(2)  # Show verification message for 2 seconds
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        if verification_successful:
            self.email_notifier.send_driver_started_notification(driver_name)
            self.current_driver = driver_name
            return True
        else:
            self.register_new_driver()
            return True
    def register_new_driver(self):
        print("\n=== New Driver Registration ===")
        name = input("Enter driver name: ")
        age = input("Enter driver age: ")
        license_no = input("Enter license number: ")
        print("\nCapturing face for registration...")
        # Capture face photo after text input
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera")
            return
        ret, frame = cam.read()
        img_path = None
        if ret and frame is not None:
            # Try crop face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
            img_path = f'drivercli_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            if len(faces):
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(img_path, face_img)
            else:
                cv2.imwrite(img_path, frame)
        cam.release()
        success = self.face_recognition.register_new_driver(name, age, license_no)
        if success:
            self.email_notifier.send_new_driver_notification(name, age, license_no, image_path=img_path)
            self.current_driver = name
            print(f"\n✓ {name} registered successfully!")
        else:
            print("\n✗ Registration failed!")
    def play_alert_sound(self):
        try:
            import winsound
            for _ in range(5):
                winsound.Beep(1000, 200)
            self.alert_played = True
        except:
            print("ALERT! ALERT! ALERT!")
    def monitor_driver(self):
        print("\n=== Driving Monitoring Active ===")
        print("Monitoring for drowsiness... Press 'q' to stop monitoring")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        alert_sent = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                drowsiness, closed_time, ann_frame = self.drowsiness_detector.detect_drowsiness(frame)
                if drowsiness and closed_time >= 3.0:
                    if not self.alert_played:
                        self.play_alert_sound()
                    if not alert_sent and self.current_driver:
                        # Save face image at drowsy state
                        drowsy_img_path = f"drowsycli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
                        if len(faces):
                            x, y, w, h = faces[0]
                            drowsy_face = frame[y:y+h, x:x+w]
                            cv2.imwrite(drowsy_img_path, drowsy_face)
                        else:
                            cv2.imwrite(drowsy_img_path, frame)
                        self.email_notifier.send_drowsiness_alert(self.current_driver, closed_time, image_path=drowsy_img_path)
                        alert_sent = True
                    cv2.putText(ann_frame, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(ann_frame, "WAKE UP!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    if closed_time < 1.0:
                        self.alert_played = False
                        alert_sent = False
                cv2.putText(ann_frame, f"Eyes closed: {closed_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if self.current_driver:
                    cv2.putText(ann_frame, f"Driver: {self.current_driver}", (ann_frame.shape[1]-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Driver Monitoring', ann_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # Loop is now continuous until 'q' is pressed
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.drowsiness_detector.reset()
    def run(self):
        print("\n" + "="*50)
        print("  DRIVER DROWSINESS DETECTION & SECURITY SYSTEM")
        print("="*50)
        while True:
            print("\n\n--- MAIN MENU ---")
            print("1. Start Driving (Verify Driver)")
            print("2. Register New Driver")
            print("3. Exit")
            choice = input("\nEnter your choice (1-3): ")
            if choice == '1':
                if self.verify_driver():
                    self.monitor_driver()
            elif choice == '2':
                self.register_new_driver()
            elif choice == '3':
                print("\nThank you for using the system!")
                break
            else:
                print("\nInvalid choice. Please try again.")
def main():
    system = DriverDrowsinessSystem()
    system.run()
if __name__ == "__main__":
    main()
