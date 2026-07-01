import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)

app.prepare(ctx_id=0)

img = cv2.imread(
    "storage/employees/EMP001/images/profile_1.jpg",
)

faces = app.get(img)

face = faces[0]

print("Available Attributes")
print(dir(face))

print()

print("Embedding Exists :", hasattr(face, "embedding"))

print("Normed Exists :", hasattr(face, "normed_embedding"))

print()

print("Embedding Norm :", np.linalg.norm(face.embedding))

if hasattr(face, "normed_embedding"):

    print(
        "Normed Embedding Norm :",
        np.linalg.norm(face.normed_embedding),
    )