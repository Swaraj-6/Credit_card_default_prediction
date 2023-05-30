import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __int__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Making data transformation object")
            # Define nominal and other features
            nominal_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
            numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            logging.info("Initiating pipeline")

            # numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('nominalencoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_cols),
                ('categorical_pipeline', categorical_pipeline, nominal_cols)
            ])

            logging.info("Pipeline completed")
            return preprocessor

        except Exception as e:
            logging.info("An exception has occurred in get_data_transformation_obj")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading train and test data completed")
            logging.info(f"Train Dataframe head : \n {train_df.head().to_string()}")
            logging.info(f"Test Dataframe head : \n {test_df.head().to_string()}")

            logging.info("Getting preprocessor object")
            preprocessor_obj = self.get_data_transformation_obj()

            target_col = 'default.payment.next.month'
            drop_cols = [target_col, 'ID']

            # Separating input and target features
            input_feature_train_df = train_df.drop(columns=drop_cols, axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=drop_cols, axis=1)
            target_feature_test_df = test_df[target_col]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("An exception has occurred in initiate_data_transformation")
            raise CustomException(e, sys)






















