import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, prefix='ts_'):
        self.data_transformation_config = DataTransformationConfig()
        self.prefix = prefix

    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        try:
            df = df.copy()
            dt = pd.to_datetime(df.index)

            original_cols = df.columns.tolist()

            # Time-based features
            df[self.prefix + 'year'] = dt.year
            df[self.prefix + 'month'] = dt.month
            df[self.prefix + 'day_of_year'] = dt.dayofyear
            df[self.prefix + 'day_of_week'] = dt.dayofweek
            df[self.prefix + 'week_of_year'] = dt.isocalendar().week.astype(int).values
            df[self.prefix + 'doy_sin'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
            df[self.prefix + 'doy_cos'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
            df[self.prefix + 'month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            df[self.prefix + 'month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            df[self.prefix + 'days_since_start'] = (dt - dt[0]).days

            # Lag and rolling features for all columns
            for col in original_cols:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_roll_mean_7d'] = df[col].rolling(7, min_periods=1).mean().shift(1)
                df[f'{col}_roll_std_7d'] = df[col].rolling(7, min_periods=1).std().shift(1)

            return df
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Phase')
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
            test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self



            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr=preprocessing_obj.fit_transform(train_df)
            test_arr=preprocessing_obj.transform(test_df)

            train_arr=train_arr.dropna()
            test_arr=test_arr.dropna()

          
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)