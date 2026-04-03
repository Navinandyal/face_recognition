import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # use -1 if no GPU

# dataset_path = r"c:\Users\navin\Downloads\Two Elephants\CP\Face rec\aligned_dataset\"
# dataset_path = "aligned_dataset"
dataset_path = "FaceRecognition"

embeddings = []
labels = []

for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) > 0:
            emb = faces[0].embedding
            embeddings.append(emb)
            labels.append(person)

embeddings = np.array(embeddings)

# Save files
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)

print("✅ Embeddings created and saved!")
print("Total faces:", len(labels))