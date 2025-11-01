import os
import sys
from sklearn 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import logging


@dataclass
class DatatransformationConfig:
    preproccessor_ob_file_path = os.path.join('artifacts', 'preprocessor')



class Data_Transformation:
    def __init__ (self, DatatransformationConfig):
        self.data_transformation_config = DatatransformationConfig()
        

    try:
        




    except Exception as e:
        raise CustomException (e, sys)



