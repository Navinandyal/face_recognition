import sys
import cv2
import numpy as np
import datetime
import time
import os
import threading
import tempfile
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from twilio.rest import Client

# =========================
# CONFIG
# =========================
CRIMINALS = ["Navin", "Sohan", "Pranav", "Akash","Shrishail"]

CAMERA_ID = "CAM-01"          # Change per camera/location
CAMERA_LOCATION = "Main Gate" # Human-readable location name

# Twilio Config
ACCOUNT_SID = "ACa3effca65a7f32b5e04bdc00babb638d"
AUTH_TOKEN  = "a6df020a6898dddbdde357d136bb74c1"
TWILIO_NUMBER = "+12603773947"

# :white_check_mark: Multiple recipients — add as many numbers as needed
ALERT_NUMBERS = [
    "+919420241898",   # Recipient 1
    "+918605083060",   # Recipient 2  ← replace with real numbers
    # "+91XXXXXXXXXX", # Add more here
]

# Twilio requires a publicly accessible URL to send MMS images.
# Upload face images here (e.g. an S3 bucket or any static file host).
# Leave as None to send SMS-only (no image).
IMAGE_HOST_URL = None  # e.g. "https://your-bucket.s3.amazonaws.com/faces/"

# =========================
# TEMP FOLDER FOR FACE IMAGES
# =========================
TEMP_FACE_DIR = os.path.join(tempfile.gettempdir(), "hvr_alerts")
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

# =========================
# SMS / MMS FUNCTION (runs in a background thread)
# =========================
def send_alert(name: str, confidence: float, face_img: np.ndarray):
    """Send MMS (with face image) or SMS to all recipients. Non-blocking."""
    def _send():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        body = (
            f":rotating_light: SECURITY ALERT :rotating_light:\n"
            f"Criminal  : {name}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Camera    : {CAMERA_ID} ({CAMERA_LOCATION})\n"
            f"Time      : {timestamp}"
        )

        # --- Save face image and get a public URL if host is configured ---
        media_url = None
        if IMAGE_HOST_URL and face_img is not None and face_img.size > 0:
            filename = f"{name}_{int(time.time())}.jpg"
            local_path = os.path.join(TEMP_FACE_DIR, filename)
            cv2.imwrite(local_path, face_img)
            media_url = IMAGE_HOST_URL.rstrip("/") + "/" + filename
            # NOTE: You must separately upload `local_path` to your host.
            # This URL tells Twilio where to fetch the image from.

        try:
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            for number in ALERT_NUMBERS:
                msg_params = dict(
                    body=body,
                    from_=TWILIO_NUMBER,
                    to=number,
                )
                if media_url:
                    msg_params["media_url"] = [media_url]

                msg = client.messages.create(**msg_params)
                print(f":white_check_mark: Alert sent to {number} | SID: {msg.sid}")

        except Exception as e:
            print(f":x: SMS Error: {e}")

    threading.Thread(target=_send, daemon=True).start()

# =========================
# Load Models
# =========================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(320, 320))

embeddings = np.load("embeddings.npy")
labels     = np.load("labels.npy")

yolo_model = YOLO("yolov8n.pt")
class_names = yolo_model.names

# =========================
# Video Thread
# =========================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    face_signal          = pyqtSignal(np.ndarray, str)
    alert_signal         = pyqtSignal(str, float)   # name + confidence

    def __init__(self):
        super().__init__()
        self.running    = True
        self.last_alert = {}   # name → last alert timestamp

    def run(self):
        # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(":x: Camera not detected")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, verbose=False)[0]

            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{class_names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                if cls == 0:  # person
                    h, w, _ = frame.shape
                    px1, py1 = max(0, x1), max(0, y1)
                    px2, py2 = min(w, x2), min(h, y2)
                    person_crop = frame[py1:py2, px1:px2]
                    if person_crop.size == 0:
                        continue

                    faces = face_app.get(person_crop)

                    for face in faces:
                        emb  = face.embedding
                        sims = cosine_similarity([emb], embeddings)[0]
                        idx  = np.argmax(sims)

                        name       = labels[idx]
                        confidence = sims[idx]

                        if confidence < 0.4:
                            name = "Unknown"

                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        fx1 += px1; fy1 += py1
                        fx2 += px1; fy2 += py1

                        color = (0, 0, 255) if name in CRIMINALS else (0, 255, 0)
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                        cv2.putText(frame,
                                    f"{name} ({confidence:.2f})",
                                    (fx1, fy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # --- ALERT LOGIC (30-second cooldown per criminal) ---
                        current_time = time.time()
                        if name in CRIMINALS:
                            if (name not in self.last_alert or
                                    current_time - self.last_alert[name] > 30):
                                self.last_alert[name] = current_time

                                face_crop = frame[fy1:fy2, fx1:fx2].copy()
                                self.alert_signal.emit(name, float(confidence))
                                # Pass face image to SMS sender
                                send_alert(name, float(confidence), face_crop)

                        face_crop = frame[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            self.face_signal.emit(face_crop.copy(), str(name))

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_img)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# =========================
# UI
# =========================
class CCTVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HVR SYSTEM")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color:#1e1e2f; color:white;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Camera label
        cam_label = QLabel(f":camera: {CAMERA_ID} — {CAMERA_LOCATION}")
        cam_label.setStyleSheet("font-size:14px; color:#aaaacc; padding:4px;")
        layout.addWidget(cam_label)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background:black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.setLayout(layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.alert_signal.connect(self.show_alert)
        self.thread.start()

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def show_alert(self, name: str, confidence: float):
        msg = QMessageBox()
        msg.setWindowTitle(":rotating_light: ALERT")
        msg.setText(
            f"Criminal Detected!\n\n"
            f"Name      : {name}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Camera    : {CAMERA_ID} ({CAMERA_LOCATION})\n"
            f"Time      : {datetime.datetime.now().strftime('%H:%M:%S')}\n\n"
            f"SMS alert sent to {len(ALERT_NUMBERS)} recipient(s)."
        )
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = CCTVApp()
    window.show()
    sys.exit(app.exec_())