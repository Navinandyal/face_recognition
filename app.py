# face rec and showing detected faces left side panel
import sys
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Load InsightFace Model
# =========================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # use -1 for CPU

# Load embeddings
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# =========================
# Video Thread
# =========================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    face_signal = pyqtSignal(np.ndarray, str)

    def run(self):
        # 👉 Change to RTSP if needed
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = face_app.get(frame)

            for face in faces:
                emb = face.embedding

                sims = cosine_similarity([emb], embeddings)[0]
                idx = np.argmax(sims)

                name = labels[idx]
                confidence = sims[idx]

                if confidence < 0.4:
                    name = "Unknown"

                x1, y1, x2, y2 = map(int, face.bbox)

                # Boundaries
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Crop face
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    self.face_signal.emit(face_crop, name)

            # Convert frame to Qt
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

            self.change_pixmap_signal.emit(qt_img)


# =========================
# Main UI
# =========================
class CCTVApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HVR SYSTEM")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color:#1e1e2f; color:white;")

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 🔝 Top Bar
        top_bar = QHBoxLayout()
        title = QLabel("📹 HVR SYSTEM")
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # 🔲 Body
        body_layout = QHBoxLayout()

        # LEFT PANEL
        # LEFT PANEL (Scrollable)
        self.face_container = QWidget()
        self.face_layout = QVBoxLayout(self.face_container)
        self.face_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Detected Faces")
        title.setStyleSheet("font-weight:bold; font-size:14px;")
        self.face_layout.addWidget(title)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.face_container)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #2e2e3e;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 4px;
            }
        """)

        body_layout.addWidget(scroll, 1)
        # self.face_layout = QVBoxLayout()
        # self.face_layout.addWidget(QLabel("Detected Faces"))

        # body_layout.addLayout(self.face_layout, 1)

        # VIDEO
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background:black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        body_layout.addWidget(self.video_label, 4)

        main_layout.addLayout(body_layout)
        self.setLayout(main_layout)

        # Thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.face_signal.connect(self.update_faces)
        self.thread.start()

    # =========================
    # Update Video
    # =========================
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        ))

    # =========================
    # Update Face Panel
    # =========================
    def update_faces(self, face_img, name):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # Convert face
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(140, 90, Qt.KeepAspectRatio)

        # Create card
        card = QWidget()
        card_layout = QVBoxLayout()

        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)

        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-weight:bold;")

        time_label = QLabel(current_time)
        time_label.setAlignment(Qt.AlignCenter)
        time_label.setStyleSheet("color:gray; font-size:10px;")

        card_layout.addWidget(img_label)
        card_layout.addWidget(name_label)
        card_layout.addWidget(time_label)

        card.setLayout(card_layout)
        card.setStyleSheet("""
            background:#2e2e3e;
            border:1px solid #444;
            padding:5px;
        """)

        # Insert at top
        self.face_layout.insertWidget(1, card)

        # Limit to 6 items
        # if self.face_layout.count() > 7:
        #     item = self.face_layout.takeAt(self.face_layout.count()-1)
        #     if item.widget():
        #         item.widget().deleteLater()


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CCTVApp()
    window.show()
    sys.exit(app.exec_())