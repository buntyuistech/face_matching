import cv2
import numpy as np
from fastapi import FastAPI
from insightface.app import FaceAnalysis

app = FastAPI()

# Load InsightFace model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1)  # CPU mode

def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise Exception(f"Cannot read image: {image_path}")

    faces = face_app.get(image)

    if len(faces) == 0:
        raise Exception("No face detected")

    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/")
def home():
    return {"status": "Server Running"}

@app.get("/verify")
def verify():

    emp = get_embedding("images/employee.jpg")
    att = get_embedding("images/attendance.jpg")

    score = float(cosine_similarity(emp, att))

    return {
        "similarity": round(score, 4),
        "matched": score > 0.85
    }