from src.face_mask_detection.logger import logging
from src.face_mask_detection.exception import CustomException
import sys
from src.face_mask_detection.components.data_validation import DataValidation
from src.face_mask_detection.components.data_ingestion import DataIngestion
from src.face_mask_detection.components.data_transformation import DataTransformation
from src.face_mask_detection.components.model_tranier import ModelTrainer
import os
print("CWD:", os.getcwd())


if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()
        data_validation = DataValidation()
        data_validation_return=data_validation.initiate_data_validation()
        print(data_validation_return)
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation()
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training()
        print("Model Trained Completed")
    except Exception as e:
            logging.error("❌ Error occurred in Data Ingestion")
            raise CustomException(e, sys)
