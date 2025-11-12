import sys
import os
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = r'artifacts/model.pkl'
            preprocessor_path = r'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 birth_year: int, 
                 potential_therapy: float,
                 employment_status: str):
        self.birth_year = birth_year
        self.potential_therapy = potential_therapy
        self.employment_status = employment_status

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'BirthYear': [self.birth_year],
                'Potential Therapy': [self.potential_therapy],
                'Employment Status': [self.employment_status]
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)
