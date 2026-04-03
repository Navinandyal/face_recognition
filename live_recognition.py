import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # use -1 if no GPU

# Load saved embeddings
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding

        sims = cosine_similarity([emb], embeddings)[0]
        idx = np.argmax(sims)

        name = labels[idx]
        confidence = sims[idx]

        # Threshold
        if confidence < 0.3:
            name = "Unknown"

        x1, y1, x2, y2 = map(int, face.bbox)

        # Fix boundaries
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Label
        text = f"{name} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()