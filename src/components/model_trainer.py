import os
import sys
from sklearn.ensemble import (
    RandomforestClassifier, logisticregression)
from catboost import CatBoostClassifier
from src.logger import logging
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join('artificats','model.pkl')



class Model_Trainer:
    def __init__ (self, ModelTrainerConfig):
        self.modeltrainerconfig = ModelTrainerConfig()