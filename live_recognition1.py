import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# ---------------- LOAD MODELS ----------------

# InsightFace (CPU safe)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU + faster

# YOLO
yolo_model = YOLO("yolov8n.pt")
class_names = yolo_model.names  # :fire: class labels

# Load embeddings
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(":x: Camera not detected")
    exit()

print("Press 'q' to quit")

frame_count = 0

# ---------------- MAIN LOOP ----------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    frame_count += 1

    # Skip frames (boost FPS)
    if frame_count % 2 != 0:
        continue

    # -------- YOLO DETECTION --------
    results = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # :fire: Draw ALL objects (not just person)
        label = f"{class_names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # -------- ONLY process PERSON for face recognition --------
        if cls == 0:  # person
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # -------- INSIGHTFACE --------
            faces = app.get(person_crop)

            for face in faces:
                emb = face.embedding

                sims = cosine_similarity([emb], embeddings)[0]
                idx = np.argmax(sims)

                name = labels[idx]
                confidence = sims[idx]

                if confidence < 0.4:
                    name = "Unknown"

                fx1, fy1, fx2, fy2 = map(int, face.bbox)

                # Adjust to original frame
                fx1 += x1
                fy1 += y1
                fx2 += x1
                fy2 += y1

                # Draw FACE box (green)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2),
                              (0, 255, 0), 2)

                text = f"{name} ({confidence:.2f})"
                cv2.putText(frame, text, (fx1, fy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()