import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DatasplitingConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    
class Dataspliting:
    def __init__(self):
        self.ingestion_config = DatasplitingConfig()
        
    def initate_data_spliting(self, df):
        logging.info('Data spliting method Started')
        try:
            logging.info('Train Test Split Initiated')

            # 80% for training, 20% for testing
            TRAIN_RATIO = 0.8
            split_point = int(len(df) * TRAIN_RATIO)

            # Training Set (First 80% of data)
            train_set = df.iloc[:split_point]

            # Testing Set (Last 20% of data)
            test_set = df.iloc[split_point:]
            

            train_set.to_csv(self.ingestion_config.train_data_path,index=True,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=True,header=True)

            logging.info('spliting of Data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data spliting stage')
            raise CustomException(e, sys)