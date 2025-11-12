import os
import sys
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join('artifacts','model.pkl')

class Model_Trainer:
    def __init__ (self, ModelTrainerConfig):
        self.modeltrainerconfig = ModelTrainerConfig()

    def  initiate_model_trainer(self, X_train, X_test, y_train, y_test, preprocessor_path):
        try:
            logging.info('Split training and test input data')

            models = {
                "Logistic regression" : LogisticRegression(),
                "Random Forest Classifier" : RandomForestClassifier(),
                "Catboost Classifier" : CatBoostClassifier()
            }

            params = {
                "Logistic regression" : {
                'C' : [0.01,0.1,1,10,100],
                 'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                 'solver' : ['liblinear', 'saga', 'lbfgs'],
                 'max_iter' : [100,200,500]
                },

                 "Random Forest Classifier" : {
                     'n_estimators' : [50,100,200,500],
                     'max_depth' : [None, 5, 10, 20],
                     'min_samples_split' : [2,5,10],
                     'min_samples_leaf' : [1,2,4],
                     'max_features' : ['sqrt', 'log2', None]
                 },
                 'CatBoost Classifier' : {
                     'depth' : [4,6,8,10],
                     'learning_rate' : [0.01, 0.055, 0.1, 0.2],
                     'iterations' : [100, 200, 500],
                     'l2_leaf_reg' : [1,3,5,7,9]
                 }

            }

            model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, params=params
            )

            # Best Model Selection and Score based of F1
            best_model_name = max(model_report, key= lambda x: model_report[x]['test_f1_score'])
            best_model_score = model_report[best_model_name]['test_f1_score']
            best_model = models[best_model_name]

            # Save best model
            save_object(file_path=self.modeltrainerconfig.trained_model_file, object=best_model)
            logging.info(f"Best model: {best_model_name} with score {best_model_score}")

            return best_model, best_model_score


        except Exception as e:
            raise CustomException(e,sys)