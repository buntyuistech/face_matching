import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------
# Add Project Root
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# ---------------------------------------------------------
# Services
# ---------------------------------------------------------

from services.face_service import FaceService
from services.storage_service import StorageService

face_service = FaceService()
storage_service = StorageService()

# ---------------------------------------------------------
# Employee
# ---------------------------------------------------------

EMPLOYEE_ID = "EMP001"

# ---------------------------------------------------------
# Test Images
# ---------------------------------------------------------
#
# Replace these with your own test images.
#
# SAME PERSON
# beard.jpg
# no_beard.jpg
# glasses.jpg
#
# DIFFERENT PERSON
# elon.jpg
# virat.jpg
# rohit.jpg
#
# ---------------------------------------------------------

TEST_IMAGES = {
    "Same Face": "tests/images/mine.jpg",
    "Beard": "tests/images/beard.jpg",
    "No Beard": "tests/images/no_beard.jpg",
    "Glasses": "tests/images/glasses.jpg",
    "Elon": "tests/images/elon.jpg",
    "Virat": "tests/images/virat.jpg",
    "Rohit": "tests/images/rohit.jpg",
}

# ---------------------------------------------------------
# Load Registered Embeddings
# ---------------------------------------------------------

registered_embeddings = storage_service.load_embeddings(
    EMPLOYEE_ID,
)

print("\n")
print("=" * 80)
print("REGISTERED EMBEDDINGS")
print("=" * 80)
print(f"Employee : {EMPLOYEE_ID}")
print(f"Profiles : {len(registered_embeddings)}")
print("=" * 80)

# ---------------------------------------------------------
# Run Tests
# ---------------------------------------------------------

for name, image_path in TEST_IMAGES.items():

    print("\n")
    print("-" * 80)
    print(name)
    print("-" * 80)

    try:

        embedding = face_service.extract_embedding_from_path(
            image_path,
        )

        result = face_service.verify_face(
            current_embedding=embedding,
            registered_embeddings=registered_embeddings,
        )

        print(f"Similarity : {result['similarity']:.4f}")
        print(f"Matched    : {result['matched']}")
        print(f"Profile    : {result['matched_index']}")

    except Exception as e:

        print("ERROR :", e)

print("\n")
print("=" * 80)
print("TEST COMPLETED")
print("=" * 80)