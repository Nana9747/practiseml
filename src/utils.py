import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.Exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_model = None
        best_score = float("-inf")

        for model_name, model in models.items():
            para = param.get(model_name, {})  # Get parameter grid for the current model

            # Perform GridSearchCV
            gs = GridSearchCV(estimator=model, param_grid=para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Use the best estimator from GridSearchCV for predictions
            fitted_model = gs.best_estimator_

            # Predictions on training and testing data
            y_train_pred = fitted_model.predict(X_train)
            y_test_pred = fitted_model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[model_name] = test_model_score

            # Track the best model based on test score
            if test_model_score > best_score:
                best_score = test_model_score
                best_model = fitted_model

        return report, best_model

    except Exception as e:
        raise CustomException(e, sys)
