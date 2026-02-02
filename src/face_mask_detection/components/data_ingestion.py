import os
import shutil
from dataclasses import dataclass
import sys

from src.face_mask_detection.logger import logging
from src.face_mask_detection.exception import CustomException


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    raw_data_dir: str = os.path.join("artifacts", "raw_data")
    images_dir: str = os.path.join("artifacts", "raw_data", "images")
    annotations_dir: str = os.path.join("artifacts", "raw_data", "annotations")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("📥 Starting Data Ingestion")

        try:
            # ✅ Explicitly create artifacts directory
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)
            logging.info("📁 Artifacts directory ensured")

            source_images_path = os.path.join("archive", "images")
            source_annotations_path = os.path.join("archive", "annotations")

            if not os.path.exists(source_images_path):
                raise FileNotFoundError("Images folder not found in archive")

            if not os.path.exists(source_annotations_path):
                raise FileNotFoundError("Annotations folder not found in archive")

            # Create raw data directories
            os.makedirs(self.ingestion_config.images_dir, exist_ok=True)
            os.makedirs(self.ingestion_config.annotations_dir, exist_ok=True)

            # Copy images
            for file_name in os.listdir(source_images_path):
                src = os.path.join(source_images_path, file_name)
                dst = os.path.join(self.ingestion_config.images_dir, file_name)
                shutil.copy(src, dst)

            # Copy annotations
            for file_name in os.listdir(source_annotations_path):
                src = os.path.join(source_annotations_path, file_name)
                dst = os.path.join(self.ingestion_config.annotations_dir, file_name)
                shutil.copy(src, dst)

            logging.info("✅ Data Ingestion completed successfully")

            return self.ingestion_config.raw_data_dir

        except Exception as e:
            logging.error("❌ Error occurred in Data Ingestion")
            raise CustomException(e, sys)
