# Basic Import
import numpy as np
import pandas as pd
import time

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts','model.pkl')
    TARGET_FEATURE: str = 'Consumption'

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models_to_test = self._define_models() # Define models once

    def _define_models(self):
        """Defines the list of models and their hyperparameter grids."""
        return [
            # ... (Model Definitions from your original code) ...
            {'name': 'XGBoost Regressor', 'estimator': xgb.XGBRegressor(random_state=42), 
             'params': {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}},
            {'name': 'Random Forest Regressor', 'estimator': RandomForestRegressor(random_state=42), 
             'params': {'n_estimators': [100, 200], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}},
            {'name': 'Ridge Regression', 'estimator': Ridge(), 
             'params': {'alpha': [0.1, 1.0, 10.0]}},
            {'name': 'Support Vector Regressor', 'estimator': SVR(), 
             'params': {'kernel': ['rbf', 'linear'], 'C': [0.1, 10], 'gamma': ['scale', 'auto']}},
            {'name': 'K-Nearest Neighbors Regressor', 'estimator': KNeighborsRegressor(), 
             'params': {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance'], 'p': [1, 2]}}
        ]

    def _train_and_evaluate_model(self, model_config, xtrain, ytrain, xtest, ytest):
        
        model_name = model_config['name']
        estimator = model_config['estimator']
        param_grid = model_config['params']

        # 1. Hyperparameter Tuning (GridSearch)
        grid_search = GridSearchCV(
            estimator=estimator, 
            param_grid=param_grid, 
            scoring='neg_mean_squared_error', 
            cv=5, 
            n_jobs=-1 
        )
        grid_search.fit(xtrain, ytrain)
        
        best_model = grid_search.best_estimator_
        
        # 2. Evaluation on Test Set
        y_pred = best_model.predict(xtest)
        
        rmse = np.sqrt(mean_squared_error(ytest, y_pred))
        mae = mean_absolute_error(ytest, y_pred)
        r2 = r2_score(ytest, y_pred)
        
        return {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Best Params': grid_search.best_params_,
            'Best Estimator': best_model
        }

    def initate_model_training(self, train_array, test_array):
        
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            TARGET_FEATURE = self.model_trainer_config.TARGET_FEATURE
            
            xtrain, ytrain, xtest, ytest = (
                train_array.drop(columns=[TARGET_FEATURE]),
                train_array[TARGET_FEATURE],
                test_array.drop(columns=[TARGET_FEATURE]),
                test_array[TARGET_FEATURE]
            )
            
            results = []
            logging.info(f"Final X_train shape: {xtrain.shape}")
            logging.info(f"Feature names: {list(xtrain.columns)}")
            
            # 1. Train and Evaluate all models
            for model_config in self.models_to_test:
                result = self._train_and_evaluate_model(model_config, xtrain, ytrain, xtest, ytest)
                results.append(result)

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by='RMSE').reset_index(drop=True)

            # 2. Select and Save the Best Model
            best_result = results_df.iloc[0]
            best_model_name = best_result['Model']
            final_model = best_result['Best Estimator'] # Use the best_estimator_ directly

            # Log metrics for the overall best model (using test set metrics from the best run)
            final_mae, final_rmse, final_r2 = best_result['MAE'], best_result['RMSE'], best_result['R2']

            logging.info(f"🏆 Best model based on RMSE: {best_model_name} with Params: {best_result['Best Params']}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )
            logging.info('Model pickle file saved')
            
            logging.info(f'Test MAE : {final_mae:.4f}')
            logging.info(f'Test RMSE : {final_rmse:.4f}')
            logging.info(f'Test R2 Score : {final_r2:.4f}')
            logging.info('Final Model Training Completed')
            
            return final_mae, final_rmse, final_r2 
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, None)
