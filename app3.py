# face rec and showing detected faces and alerting on criminals with whatsapp (Twilio) and image upload (Cloudinary)
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

# ✅ NEW (for image upload)
import cloudinary
import cloudinary.uploader

# =========================
# CONFIG
# =========================
CRIMINALS = ["Navin", "Sohan", "Pranav", "Akash","Shrishail"]

CAMERA_ID = "CAM-01"
CAMERA_LOCATION = "Main Gate"

# Twilio Config
ACCOUNT_SID = "ACc8982b79e38ee7ae94d7f0b1889910c2"
AUTH_TOKEN  = "2b00fdd2079212ae4cd03e23d88c37b7"

# WhatsApp numbers (+91 works here)
ALERT_NUMBERS = [
    "+919405694928",
    "+918605083060",
]

# ✅ Cloudinary Config
cloudinary.config(
    cloud_name="dd4yq2pwq",
    api_key="975589958339315",
    api_secret="v8rSOxWL9f4B62fLpr0SgRX2Q5M"
)

# =========================
# TEMP FOLDER
# =========================
TEMP_FACE_DIR = os.path.join(tempfile.gettempdir(), "hvr_alerts")
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

# =========================
# WHATSAPP ALERT FUNCTION
# =========================
def send_alert(name: str, confidence: float, face_img: np.ndarray):

    def _send():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        body = (
            f"🚨 SECURITY ALERT 🚨\n"
            f"Criminal  : {name}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Camera    : {CAMERA_ID} ({CAMERA_LOCATION})\n"
            f"Time      : {timestamp}"
        )

        media_url = None
        local_path = None

        try:
            # Save temp image
            filename = f"{name}_{int(time.time())}.jpg"
            local_path = os.path.join(TEMP_FACE_DIR, filename)

            face_img_small = cv2.resize(face_img, (300, 300))
            cv2.imwrite(local_path, face_img_small)

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(local_path)
            media_url = upload_result["secure_url"]

        except Exception as e:
            print("❌ Upload Error:", e)

        try:
            client = Client(ACCOUNT_SID, AUTH_TOKEN)

            for number in ALERT_NUMBERS:
                msg = client.messages.create(
                    body=body,
                    from_='whatsapp:+14155238886',
                    to=f'whatsapp:{number}',
                    media_url=[media_url] if media_url else None
                )

                print(f"✅ WhatsApp sent to {number} | SID: {msg.sid}")

        except Exception as e:
            print("❌ WhatsApp Error:", e)

        finally:
            # Delete temp image
            if local_path and os.path.exists(local_path):
                os.remove(local_path)

    threading.Thread(target=_send, daemon=True).start()

# =========================
# LOAD MODELS
# =========================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(320, 320))

embeddings = np.load("embeddings.npy")
labels     = np.load("labels.npy")

yolo_model = YOLO("yolov8n.pt")
class_names = yolo_model.names

# =========================
# VIDEO THREAD
# =========================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    face_signal          = pyqtSignal(np.ndarray, str)
    alert_signal         = pyqtSignal(str, float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.last_alert = {}

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Camera not detected")
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

                if cls == 0:  # person
                    person_crop = frame[y1:y2, x1:x2]
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
                        fx1 += x1; fy1 += y1
                        fx2 += x1; fy2 += y1

                        color = (0, 0, 255) if name in CRIMINALS else (0, 255, 0)

                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                        cv2.putText(frame,
                                    f"{name} ({confidence:.2f})",
                                    (fx1, fy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # ALERT (30 sec cooldown)
                        current_time = time.time()
                        if name in CRIMINALS and confidence > 0.5:
                            if (name not in self.last_alert or
                                    current_time - self.last_alert[name] > 30):

                                self.last_alert[name] = current_time

                                face_crop = frame[fy1:fy2, fx1:fx2].copy()
                                self.alert_signal.emit(name, float(confidence))
                                send_alert(name, float(confidence), face_crop)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        cam_label = QLabel(f"📷 {CAMERA_ID} — {CAMERA_LOCATION}")
        cam_label.setStyleSheet("font-size:14px; color:#aaaacc;")
        layout.addWidget(cam_label)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.setLayout(layout)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.alert_signal.connect(self.show_alert)
        self.thread.start()

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def show_alert(self, name, confidence):
        QMessageBox.critical(
            self,
            "🚨 ALERT",
            f"Criminal Detected!\n\n{name} ({confidence:.1%})"
        )

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CCTVApp()
    window.show()
    sys.exit(app.exec_())