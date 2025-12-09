import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from database import DriverDatabase
from face_recognition_simple import SimpleFaceRecognition
from drowsiness_detection import DrowsinessDetector
from email_notifier import EmailNotifier
import pickle
import os

st.set_page_config(page_title="Driver Drowsiness & Security", layout="wide")

# Ensure directories for storing images
REGISTER_DIR = "registered_drivers"
DROWSY_DIR = "drowsy_events"
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(DROWSY_DIR, exist_ok=True)

db = DriverDatabase()
face_recog = SimpleFaceRecognition(db)
drowse = DrowsinessDetector()
notifier = EmailNotifier()

def validate_registration(name: str, age: int, license_no: str):
    if not name or not name.replace(' ', '').isalpha():
        return False, "Name must contain only letters."
    if not (18 <= age <= 80):
        return False, "Age must be between 18 and 80."
    if not (license_no.isdigit() and len(license_no) == 6):
        return False, "License number must be exactly 6 digits."
    return True, None

def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

st.title("Driver Drowsiness Detection & Security - Streamlit")

tab1, tab2, tab3 = st.tabs(["Register Driver", "Start Driving", "All Drivers"])

with tab1:
    st.header("Register New Driver")
    is_adult_select = st.selectbox("Are you 18+?", ["No", "Yes"])
    is_adult = is_adult_select == "Yes"
    
    if not is_adult:
        st.error("You can't drive.")
    else:
        name = st.text_input("Name (letters only)")
        age = st.number_input("Age (18-80)", min_value=18, max_value=80, value=18, step=1)
        license_no = st.text_input("License Number (6 digits)")
        capture_btn = st.button("Capture & Register")

        if capture_btn:
            valid, msg = validate_registration(name, int(age), license_no)
            if not valid:
                st.error(msg)
            else:
                st.info("Capturing image from camera...")
                frame = capture_image_from_camera()
                if frame is None:
                    st.error("Could not access camera.")
                else:
                    samples = face_recog.capture_face_for_training()
                    if not samples or len(samples) < 5:
                        st.error("Failed to capture enough face samples. Try again.")
                    else:
                        encoded_face = pickle.dumps(samples)
                        driver_id = db.add_driver(name, int(age), license_no, encoded_face)
                        if driver_id:
                            face_recog.load_recognizer()
                            st.success(f"Driver {name} registered successfully!")
                            # Save image and send email with attachment (cropped face) into REGISTER_DIR
                            img_path = os.path.join(REGISTER_DIR, f"driver_{driver_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                            if frame is not None:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
                                if len(faces):
                                    x, y, w, h = faces[0]
                                    face_img = frame[y:y+h, x:x+w]
                                    cv2.imwrite(img_path, face_img)
                                else:
                                    cv2.imwrite(img_path, frame)
                            notifier.send_new_driver_notification(name, age, license_no, image_path=img_path)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(rgb, caption="Captured Face", use_container_width=True)
                        else:
                            st.error("Registration failed (license may already exist).")

with tab2:
    st.header("Start Driving")
    status_placeholder = st.empty()
    image_placeholder = st.empty()
    monitor_placeholder = st.empty()
    
    if "verified_driver" not in st.session_state:
        st.session_state.verified_driver = None
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False

    if st.button("Verify Driver"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera.")
        else:
            status_placeholder.info("Position your face in front of the camera...")
            current_driver = None
            for _ in range(300):  # ~10 seconds at 30 FPS
                ret, frame = cap.read()
                if not ret:
                    break
                name = face_recog.recognize_face_in_frame(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(rgb, caption="Verifying...", use_container_width=True)
                if name:
                    current_driver = name
                    break
                time.sleep(0.03)
            cap.release()
            if current_driver:
                st.session_state.verified_driver = current_driver
                status_placeholder.success(f"Verified: {current_driver}")
            else:
                status_placeholder.error("Driver not recognized. Please register first.")
    if st.session_state.verified_driver:
        if not st.session_state.monitoring:
            start = monitor_placeholder.button("Start Monitoring")
            if start:
                st.session_state.monitoring = True
        else:
            stop = monitor_placeholder.button("Stop Monitoring")
            if stop:
                st.session_state.monitoring = False
        if st.session_state.monitoring:
            cap = cv2.VideoCapture(0)
            alert_sent = False
            status_placeholder.info("Monitoring for drowsiness... (Press Stop Monitoring to end)")
            drowse.reset()
            while st.session_state.monitoring:
                ret, frame = cap.read()
                if not ret:
                    break
                drowsy, closed_time, annotated = drowse.detect_drowsiness(frame)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                image_placeholder.image(rgb, caption=f"Eyes closed: {closed_time:.2f}s", use_container_width=True)
                if drowsy and closed_time >= 3.0 and not alert_sent:
                    # Play a beep sound on Windows (server-side)
                    try:
                        import winsound
                        for _ in range(5):
                            winsound.Beep(1000, 200)
                    except Exception:
                        pass
                    # Save drowsy face at moment and attach to alert email - into DROWSY_DIR
                    drowsy_img_path = os.path.join(DROWSY_DIR, f"drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
                    if len(faces):
                        x, y, w, h = faces[0]
                        drowsy_face = frame[y:y+h, x:x+w]
                        cv2.imwrite(drowsy_img_path, drowsy_face)
                    else:
                        cv2.imwrite(drowsy_img_path, frame)
                    notifier.send_drowsiness_alert(st.session_state.verified_driver, closed_time, image_path=drowsy_img_path)
                    alert_sent = True
                time.sleep(0.03)
                if not st.session_state.monitoring:
                    break
            cap.release()
            status_placeholder.info("Monitoring finished.")

with tab3:
    st.header("All Registered Drivers")
    drivers = db.get_all_drivers()

    if "edit_driver_id" not in st.session_state:
        st.session_state["edit_driver_id"] = None
    if "delete_confirm_id" not in st.session_state:
        st.session_state["delete_confirm_id"] = None

    cols = st.columns([1,2,1,2,2,4,1,1])
    cols[0].markdown("**ID**")
    cols[1].markdown("**Name**")
    cols[2].markdown("**Age**")
    cols[3].markdown("**License**")
    cols[4].markdown("**Registered At**")
    cols[5].markdown("**Face Sample**")
    cols[6].markdown("")
    cols[7].markdown("")
    for driver in drivers:
        id, name, age, license_no, face_encoding, registered_at = driver[0], driver[1], driver[2], driver[3], driver[4], driver[5]
        cols = st.columns([1,2,1,2,2,4,1,1])

        in_edit_mode = st.session_state["edit_driver_id"] == id
        in_delete_mode = st.session_state["delete_confirm_id"] == id
        if in_edit_mode:
            new_name = cols[1].text_input(f"Name_{id}", value=name)
            new_age = cols[2].number_input(f"Age_{id}", 18, 80, value=age)
            new_license = cols[3].text_input(f"License_{id}", value=license_no)
            cols[0].write(id)
            cols[4].write(str(registered_at))
            try:
                import pickle as _p
                import numpy as _np
                import PIL.Image as _PIL
                decoded = _p.loads(face_encoding)
                face_img = decoded[0] if isinstance(decoded, list) and len(decoded) else None
                if face_img is not None:
                    arr = _np.array(face_img, dtype=_np.uint8)
                    img = _PIL.Image.fromarray(arr)
                    cols[5].image(img, width=80)
                else:
                    cols[5].write("No image")
            except Exception:
                cols[5].write("Image Error")
            save_btn = cols[6].button("Save", key=f"save_{id}")
            cancel_btn = cols[7].button("Cancel", key=f"cancel_{id}")
            if save_btn:
                update_driver_sql = f"UPDATE drivers SET name = ?, age = ?, license_no = ? WHERE id = ?"
                import sqlite3
                with sqlite3.connect(db.db_path) as conn:
                    conn.execute(update_driver_sql, (new_name, int(new_age), new_license, id))
                    conn.commit()
                st.session_state["edit_driver_id"] = None
                st.rerun()
            if cancel_btn:
                st.session_state["edit_driver_id"] = None
                st.rerun()
        elif in_delete_mode:
            cols[1].write(f"Are you sure you want to delete {name}?")
            confirm_btn = cols[6].button("Yes, delete", key=f"confirmdel_{id}")
            cancel_del_btn = cols[7].button("Cancel", key=f"canceldel_{id}")
            if confirm_btn:
                import sqlite3
                with sqlite3.connect(db.db_path) as conn:
                    conn.execute("DELETE FROM drivers WHERE id = ?", (id,))
                    conn.commit()
                st.session_state["delete_confirm_id"] = None
                st.rerun()
            if cancel_del_btn:
                st.session_state["delete_confirm_id"] = None
                st.rerun()
        else:
            cols[0].write(id)
            cols[1].write(name)
            cols[2].write(age)
            cols[3].write(license_no)
            cols[4].write(str(registered_at))
            try:
                import pickle as _p
                import numpy as _np
                import PIL.Image as _PIL
                decoded = _p.loads(face_encoding)
                face_img = decoded[0] if isinstance(decoded, list) and len(decoded) else None
                if face_img is not None:
                    arr = _np.array(face_img, dtype=_np.uint8)
                    img = _PIL.Image.fromarray(arr)
                    cols[5].image(img, width=80)
                else:
                    cols[5].write("No image")
            except Exception:
                cols[5].write("Image Error")
            edit_btn = cols[6].button("Edit", key=f"edit_{id}")
            delete_btn = cols[7].button("Delete", key=f"del_{id}")
            if edit_btn:
                st.session_state["edit_driver_id"] = id
                st.rerun()
            if delete_btn:
                st.session_state["delete_confirm_id"] = id
                st.rerun()


