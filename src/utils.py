import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import accuracy_score,f1_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object (file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name,model_obj in models.items():
            
            # Select Paramter Grid
            param_grid = params[model_name]

            # Grid Search for the best hyperparameters
            gs = GridSearchCV(model_obj, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Get best models with tuned params
            best_model = gs.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Store results in report dictionary
            report[model_name] = {
                'best_params': gs.best_params_,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_f1_score': train_f1,
                'test_f1_score': test_f1
                }


            logging.info("Model Evaluation Complete")
            return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                return dill.load(file_obj)
            
        except Exception as e:
            raise CustomException(e,sys)







