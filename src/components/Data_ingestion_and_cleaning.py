import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.Data_spliting import Dataspliting, DatasplitingConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            PowerGen = pd.read_csv("notebook/data/Daily Power Generation Reports.csv") 
            CoalStock = pd.read_csv("notebook/data/Daily Coal Stock Report.csv") 
            logging.info('Datasets read as pandas Dataframe')
            
            PowerGen = PowerGen.drop(columns=['Country','Year', 'State Or Plant Name','Classification Of Plants', 'Mode Of Transport','Name Of Thermal Power Station Or Performance Of Utility', 'Type Of Sector','Name Of The Utility'])
            CoalStock = CoalStock.drop(columns=['Country','Year','Month','Type Of Category'])
            PowerGen = PowerGen.rename(columns={'Normative Stock Required (UOM:t(Tonne)), Scaling Factor:1000': 'Normative Stock Required', 'Actual Indigenous Stock (UOM:t(Tonne)), Scaling Factor:1000': 'Indigenous Stock','Actual Import Stock (UOM:t(Tonne)), Scaling Factor:1000': 'Import', 'Receipt Of The Day (UOM:t(Tonne)), Scaling Factor:1000': 'Receipt','Consumption Of The Day (UOM:t(Tonne)), Scaling Factor:1000': 'Consumption'})
            CoalStock = CoalStock.rename(columns={'Monitored Capacity  (UOM:MW(MegaWatt)), Scaling Factor:1': 'Monitored Capacity(coal)','Targeted Generation Of Electricity For The Current Day (UOM:MU(MillionUnits)), Scaling Factor:1': 'Electricity Production Target(coal)','Actual Generation Of Electricity For The Current Day (UOM:MU(MillionUnits)), Scaling Factor:1': 'Electricity Generated'})
            PowerGen['Calendar Day'] = pd.to_datetime(PowerGen['Calendar Day'])
            CoalStock['Calendar Day'] = pd.to_datetime(CoalStock['Calendar Day'])
            PowerGen = PowerGen.groupby('Calendar Day').sum().reset_index().sort_values('Calendar Day', ascending=False)
            

            fuel_dfs = {fuel: CoalStock[CoalStock['Types Of Fuel'] == fuel].copy() for fuel in CoalStock['Types Of Fuel'].unique()}
            coal = fuel_dfs['Coal']
            Nuclear = fuel_dfs['Nuclear']
            Hydro = fuel_dfs['Hydro']

            coal = coal.drop(columns=['Types Of Fuel'])
            Nuclear = Nuclear.drop(columns=['Types Of Fuel','Monitored Capacity(coal)','Electricity Production Target(coal)'])
            Hydro = Hydro.drop(columns=['Types Of Fuel','Monitored Capacity(coal)','Electricity Production Target(coal)'])
            
            CoalStock = coal.merge(Nuclear, on='Calendar Day', suffixes=('_Coal', '_Nuclear')) \
                .merge(Hydro, on='Calendar Day') \
                .rename(columns={
                    'Electricity Generated': 'Electricity Generated_Hydro'
                })
            
            df = PowerGen.merge(CoalStock, on='Calendar Day')

            df = df[(df['Consumption'] != 0) & (df['Consumption'] < 3000)].copy()
            df = df.set_index('Calendar Day')

            columns_to_drop = ['Indigenous Stock', 'Import','Electricity Production Target(coal)']
            df = df.drop(columns=columns_to_drop, axis=1)
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=True)
            
            

            return(df)

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    df = obj.initate_data_ingestion()

    data_split = Dataspliting()
    train_data, test_data = data_split.initate_data_spliting(df)

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    modeltrainer.initate_model_training(train_arr, test_arr)