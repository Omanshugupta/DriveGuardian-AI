# 🚗 DriveGuardian-AI

Python-based driver authentication and drowsiness detection for safer driving. Runs as a CLI or Streamlit app using OpenCV LBPH face recognition, Haar cascades for eye detection, and email alerts.

## 🔥 Key Features

### 👤 **Driver Authentication**
- AI-based facial recognition using LBPH (OpenCV)
- Secure driver registration system with:
  - Name validation  
  - Age verification (18+)  
  - Unique 6-digit license authentication  
- Stores face samples directly inside SQLite database
- Sends email notification whenever a new driver is registered

### 😴 **Real-Time Drowsiness Detection**
- Detects eye closure using Haar Cascade + EAR logic
- Triggers alert if eyes stay closed for **3+ seconds**
- Plays loud beeping warning (Windows compatible)
- Sends automatic drowsiness alert email with a snapshot

### 🔐 **Vehicle Security**
- Only verified drivers can start driving
- Complete event logging and email notifications
- Unauthorized drivers are immediately blocked & flagged

### 🖥️ **Streamlit Web Dashboard**
- Register drivers with live camera capture  
- Start driving mode with verification workflow  
- Real-time drowsiness monitoring  
- Edit/Delete drivers directly from the UI  
- Displays captured face images in the table


## Requirements
- Python 3.7+ (tested with 3.11)
- Webcam/camera
- Internet connection for email alerts
- SMTP account (Gmail with app password recommended)

## Dependencies
Key packages (`requirements.txt`):
- opencv-contrib-python
- numpy
- Pillow
- streamlit

Uses OpenCV’s built-in LBPH recognizer (no dlib or deep models).

## ⚙️ Setup
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Configure email 
- Copy `config.example.json` to `config.json` and fill in SMTP settings.  
- For Gmail: create an App Password and use it as `sender_password`.



## 🚀 Running the System
### 🖥️ Run CLI Version
```bash
python main.py
```

### 🌐 Run Streamlit App
```bash
streamlit run app.py
```

## How It Works
1. **Driver auth**: LBPH model trains on face crops stored in SQLite (pickled arrays).  
2. **Verification**: Captures a frame, predicts label; if unknown, prompts registration.  
3. **Drowsiness**: Haar eye detection + heuristic EAR estimate; triggers after ~3s closed eyes, plays beeps, emails alert with snapshot.  
4. **Notifications**: SMTP emails for registration, start driving, and drowsiness events.

## Project Structure
- `main.py` — CLI flow (verify/register, monitor, alerts)
- `app.py` — Streamlit UI (register, start driving, manage drivers)
- `database.py` — SQLite wrapper for drivers and face samples
- `face_recognition_simple.py` — LBPH training/prediction helpers
- `drowsiness_detection.py` — Eye-closure heuristic with Haar cascades
- `email_notifier.py` — SMTP email helpers
- `requirements.txt` — Python deps
- `config.example.json` — Sample config (create `config.json` locally)

## Customization
- Drowsiness sensitivity (`drowsiness_detection.py`):
  - `EAR_THRESHOLD` (lower = more sensitive)
  - `CLOSED_DURATION_SECONDS` (time to trigger alert)
- Alert sound: adjust `play_alert_sound()` in `main.py` / beep loop in `app.py`
- Database path: change default in `DriverDatabase` (`database.py`)

## Troubleshooting
- Camera not opening: ensure free device; try indices 0/1/2 in `VideoCapture`.
- Poor recognition: improve lighting, center face, remove glare/glasses if possible, capture more samples.
- Email failures: recheck `config.json`, internet, SMTP/app password validity.

## Security Notes
- Never commit `config.json`, databases, or face images.
- Use app passwords; rotate credentials if exposed.
- For production, consider encryption and stricter access controls.

## Roadmap
- Better landmark-based EAR
- Web/REST API
- Mobile/edge deployment
- Voice alerts

---
Drive safe! 🚗💨
