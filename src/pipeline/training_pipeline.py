import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass

    def model_train(self):
        try:
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            model_trainer = ModelTrainer()
            best_model, best_model_score = model_trainer.initiate_model_training(train_arr, test_arr)

            return best_model, best_model_score

        except Exception as e:
            logging.info("Error occurred in model_training method")
            raise CustomException(e, sys)






if __name__ == '__main__':
    obj = TrainingPipeline()
    obj.model_train()


