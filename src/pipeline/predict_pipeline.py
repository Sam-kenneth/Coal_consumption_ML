import sys
import pandas as pd
import numpy as np
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.history_data_path = os.path.join('artifacts', 'data.csv')
        

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            history_df = pd.read_csv(self.history_data_path, index_col=0, parse_dates=True)
            history_df.loc[features.index[0]] = features.iloc[0]
            df_recent = history_df.sort_index().tail(8)

            df_transformed = preprocessor.transform(df_recent)
            X = df_transformed.drop(columns=['Consumption'], errors='ignore')
            X_clean = X.dropna()
            if X_clean.empty:
                 raise ValueError("Insufficient history or invalid data for transformation. Check the date or history file.")
            X_pred = X_clean.loc[features.index[0]].values.reshape(1, -1)
            pred = model.predict(X_pred)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 date: str, # New: Date for prediction (YYYY-mm-dd)
                 NSR: float,
                 Receipt: float,
                 MC: float,
                 EC: float,
                 EN: float,
                 EH: float):
        
        self.date = pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')
        self.NSR = NSR
        self.Receipt = Receipt
        self.MC = MC
        self.EC = EC
        self.EN = EN
        self.EH = EH

    def get_data_as_dataframe(self):
        """Converts collected inputs into a DataFrame row, indexed by date."""
        try:
            if pd.isna(self.date):
                raise ValueError("Invalid date format. Use YYYY-mm-dd.")
                
            custom_data_input_dict = {
                'Normative Stock Required': [self.NSR],
                'Receipt': [self.Receipt],
                'Consumption': [np.nan], # Target is unknown for prediction
                'Monitored Capacity(coal)': [self.MC],
                'Electricity Generated_Coal': [self.EC],
                'Electricity Generated_Nuclear': [self.EN],
                'Electricity Generated_Hydro': [self.EH]
            }
            # Create DataFrame and set the date as the index
            df = pd.DataFrame(custom_data_input_dict, index=[self.date])
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
            