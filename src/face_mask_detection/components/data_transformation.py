import os
import shutil
import random
import sys
from dataclasses import dataclass

from src.face_mask_detection.logger import logging
from src.face_mask_detection.exception import CustomException


@dataclass
class DataTransformationConfig:
    # Raw data paths
    raw_images_dir: str = os.path.join("artifacts", "raw_data", "images")
    raw_annotations_dir: str = os.path.join("artifacts", "raw_data", "annotations")

    # Transformed data base dir
    transformed_dir: str = os.path.join("artifacts", "transformed_data")

    # Train paths
    train_images_dir: str = os.path.join(
        "artifacts", "transformed_data", "train", "images"
    )
    train_annotations_dir: str = os.path.join(
        "artifacts", "transformed_data", "train", "annotations"
    )

    # Validation paths
    val_images_dir: str = os.path.join(
        "artifacts", "transformed_data", "val", "images"
    )
    val_annotations_dir: str = os.path.join(
        "artifacts", "transformed_data", "val", "annotations"
    )

    train_split: float = 0.8


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self):
        logging.info("🔄 Starting Data Transformation")

        try:
            # 1️⃣ Create required directories
            os.makedirs(self.config.train_images_dir, exist_ok=True)
            os.makedirs(self.config.train_annotations_dir, exist_ok=True)
            os.makedirs(self.config.val_images_dir, exist_ok=True)
            os.makedirs(self.config.val_annotations_dir, exist_ok=True)

            # 2️⃣ Read all images
            images = os.listdir(self.config.raw_images_dir)

            if len(images) == 0:
                raise ValueError("No images found in raw images directory")

            random.shuffle(images)

            # 3️⃣ Train / Validation split
            split_index = int(len(images) * self.config.train_split)
            train_images = images[:split_index]
            val_images = images[split_index:]

            # 4️⃣ Copy TRAIN data
            for img in train_images:
                src_img = os.path.join(self.config.raw_images_dir, img)
                dst_img = os.path.join(self.config.train_images_dir, img)
                shutil.copy(src_img, dst_img)

                # Robust annotation mapping (works for .png, .jpg, .jpeg)
                ann_file = os.path.splitext(img)[0] + ".xml"
                src_ann = os.path.join(self.config.raw_annotations_dir, ann_file)
                dst_ann = os.path.join(self.config.train_annotations_dir, ann_file)

                if not os.path.exists(src_ann):
                    raise FileNotFoundError(f"Annotation missing for image: {img}")

                shutil.copy(src_ann, dst_ann)

            # 5️⃣ Copy VALIDATION data
            for img in val_images:
                src_img = os.path.join(self.config.raw_images_dir, img)
                dst_img = os.path.join(self.config.val_images_dir, img)
                shutil.copy(src_img, dst_img)

                ann_file = os.path.splitext(img)[0] + ".xml"
                src_ann = os.path.join(self.config.raw_annotations_dir, ann_file)
                dst_ann = os.path.join(self.config.val_annotations_dir, ann_file)

                if not os.path.exists(src_ann):
                    raise FileNotFoundError(f"Annotation missing for image: {img}")

                shutil.copy(src_ann, dst_ann)

            logging.info("✅ Data Transformation completed successfully")
            return self.config.transformed_dir

        except Exception as e:
            logging.error("❌ Error occurred during Data Transformation")
            raise CustomException(e, sys)
