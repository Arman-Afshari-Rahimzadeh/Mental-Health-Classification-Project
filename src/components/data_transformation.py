import os
import sys
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.utils import save_object
from src.exception import CustomException
import pandas as pd
import numpy as np
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Preprocessing pipeline for features
    def get_transformer_object(self):
        try:
            num_columns = ['BirthYear', 'Potential Therapy']
            cat_columns = ['Employment Status']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, num_columns),
                    ('categorical_pipeline', cat_pipeline, cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Main function to transform data
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read cleaned train/test data
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)
            logging.info("Read train and test data completed")

            # Get preprocessing object
            preprocessor_obj = self.get_transformer_object()

            target_column = 'Stress Category'

            # Separate features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # Encode target labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            # Save preprocessor and LabelEncoder
            os.makedirs('artifacts', exist_ok=True)
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)
            joblib.dump(le, self.data_transformation_config.label_encoder_file_path)

            logging.info("Applied preprocessing on features and encoded target labels")

            # Apply preprocessing to features
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            # Return separately
            return X_train_arr, X_test_arr, y_train_encoded, y_test_encoded, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)













