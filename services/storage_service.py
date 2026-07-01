import json
import shutil
from pathlib import Path
from typing import List

import numpy as np


class StorageService:

    STORAGE_PATH = Path("storage/employees")

    # ==========================================================
    # Employee
    # ==========================================================

    def employee_folder(
        self,
        employee_id: str,
    ) -> Path:

        return self.STORAGE_PATH / employee_id

    def employee_exists(
        self,
        employee_id: str,
    ) -> bool:

        return self.employee_folder(
            employee_id,
        ).exists()

    def create_employee(
        self,
        employee_id: str,
    ):

        folder = self.employee_folder(
            employee_id,
        )

        (folder / "images").mkdir(
            parents=True,
            exist_ok=True,
        )

        (folder / "embeddings").mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Images
    # ==========================================================

    def save_profile_image(
        self,
        employee_id: str,
        image_path: str,
        index: int,
    ):

        destination = (
            self.employee_folder(employee_id)
            / "images"
            / f"profile_{index}.jpg"
        )

        shutil.copy2(
            image_path,
            destination,
        )

    def load_profile_images(
        self,
        employee_id: str,
    ) -> List[Path]:

        folder = (
            self.employee_folder(employee_id)
            / "images"
        )

        return sorted(
            folder.glob(
                "profile_*.jpg",
            )
        )

    # ==========================================================
    # Embeddings
    # ==========================================================

    def save_embedding(
        self,
        employee_id: str,
        embedding: np.ndarray,
        index: int,
    ):

        destination = (
            self.employee_folder(employee_id)
            / "embeddings"
            / f"embedding_{index}.npy"
        )

        np.save(
            destination,
            embedding.astype(
                np.float32,
            ),
        )

    def load_embeddings(
        self,
        employee_id: str,
    ) -> List[np.ndarray]:

        folder = (
            self.employee_folder(employee_id)
            / "embeddings"
        )

        embeddings = []

        files = sorted(
            folder.glob(
                "embedding_*.npy",
            )
        )

        for file in files:

            embedding = np.load(
                file,
            ).astype(
                np.float32,
            )

            embeddings.append(
                embedding,
            )

        return embeddings

    # ==========================================================
    # Metadata
    # ==========================================================

    def save_metadata(
        self,
        employee_id: str,
        total_profiles: int,
        has_glasses_profile: bool,
    ):

        metadata = {
            "employeeId": employee_id,
            "totalProfiles": total_profiles,
            "hasGlassesProfile": has_glasses_profile,
        }

        metadata_path = (
            self.employee_folder(employee_id)
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

    def load_metadata(
        self,
        employee_id: str,
    ):

        metadata_path = (
            self.employee_folder(employee_id)
            / "metadata.json"
        )

        if not metadata_path.exists():
            return None

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file,
            )

    # ==========================================================
    # Delete Employee
    # ==========================================================

    def delete_employee(
        self,
        employee_id: str,
    ):

        folder = self.employee_folder(
            employee_id,
        )

        if folder.exists():
            shutil.rmtree(
                folder,
            )