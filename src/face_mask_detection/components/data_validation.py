import os
from dataclasses import dataclass
import sys

from src.face_mask_detection.logger import logging
from src.face_mask_detection.exception import CustomException

@dataclass
class DataValidationConfig:
    image_dir :str = os.path.join("artifacts","raw_data","images")
    annotations_dir:str = os.path.join("artifacts","raw_data","annotations")
class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()
    
    def initiate_data_validation(self):
        logging.info("Data validation started")
        try:
            if not os.path.exists(self.validation_config.image_dir):
                raise FileNotFoundError("Image file is not exists")
            if not os.path.exists(self.validation_config.annotations_dir):
                raise FileNotFoundError("Annotation file is not exists")
            image_files = os.listdir(self.validation_config.image_dir)
            annotations_files = os.listdir(self.validation_config.annotations_dir)

            if len(image_files) == 0:
                raise ValueError("Image file is empty")
            if len(annotations_files) == 0:
                raise ValueError("Annotation file is  empty")
            if len(image_files) != len(annotations_files):
                raise ValueError(
                    f"Mismatch: {len(image_files)} images but {len(annotations_files)} annotations"
                )
            return True
        except Exception as e:
            logging.error("❌ Error occurred during Data Validation")
            raise CustomException(e, sys)