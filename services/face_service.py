from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from core.constants import (
    DETECTION_SIZE,
    FACE_MODEL_NAME,
    MATCH_THRESHOLD,
    MODEL_PROVIDER,
    Messages,
)


class FaceService:

    def __init__(self):

        self.app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=[MODEL_PROVIDER],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=DETECTION_SIZE,
        )

    # ==========================================================
    # Load Image
    # ==========================================================

    def load_image(
        self,
        image_path: str,
    ) -> np.ndarray:

        image = cv2.imread(image_path)

        if image is None:
            raise Exception(
                "Unable to read image."
            )

        return image

    # ==========================================================
    # Detect Single Face
    # ==========================================================

    def detect_face(
        self,
        image: np.ndarray,
    ):

        faces = self.app.get(image)

        if len(faces) == 0:
            raise Exception(
                Messages.NO_FACE,
            )

        if len(faces) > 1:
            raise Exception(
                Messages.MULTIPLE_FACES,
            )

        return faces[0]

    # ==========================================================
    # Extract Normalized Embedding
    # ==========================================================

    def extract_embedding(
        self,
        face,
    ) -> np.ndarray:

        embedding = np.asarray(
            face.normed_embedding,
            dtype=np.float32,
        )

        return embedding

    # ==========================================================
    # Extract Embedding From Image
    # ==========================================================

    def extract_embedding_from_path(
        self,
        image_path: str,
    ) -> np.ndarray:

        image = self.load_image(
            image_path,
        )

        face = self.detect_face(
            image,
        )

        return self.extract_embedding(
            face,
        )

    # ==========================================================
    # Compare Two Embeddings
    # ==========================================================

    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:

        similarity = float(
            np.dot(
                embedding1,
                embedding2,
            )
        )

        return similarity

    # ==========================================================
    # Verify Face
    # ==========================================================

    def verify_face(
        self,
        current_embedding: np.ndarray,
        registered_embeddings: List[np.ndarray],
    ):

        best_similarity = -1.0
        best_index = -1

        print("\n")
        print("=" * 60)
        print("FACE MATCHING")
        print("=" * 60)

        for index, registered in enumerate(
            registered_embeddings,
            start=1,
        ):

            similarity = self.compare_embeddings(
                current_embedding,
                registered,
            )

            print(
                f"Profile {index} : {similarity:.4f}"
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        matched = (
            best_similarity >= MATCH_THRESHOLD
        )

        print("-" * 60)
        print(
            f"Best Similarity : {best_similarity:.4f}"
        )
        print(
            f"Threshold       : {MATCH_THRESHOLD}"
        )
        print(
            f"Matched         : {matched}"
        )
        print("=" * 60)

        return {
            "matched": matched,
            "similarity": round(
                best_similarity,
                4,
            ),
            "threshold": MATCH_THRESHOLD,
            "matched_index": best_index,
        }