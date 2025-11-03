import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'mental_health_cleaned.xlsx')

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            mental_health_survey_df = pd.read_excel("Cleaned Data/mental_health_df.xlsx")
            logging.info('Read the Excel Data')

            # Save raw data
            mental_health_survey_df.to_excel(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved successfully')

            # Train Test Split
            train_set, test_set = train_test_split(mental_health_survey_df, test_size=0.2, random_state=42)

            # Save Train and Test Sets
            train_set.to_excel(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_excel(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train and Test sets saved successfully')

        except Exception as e:
            raise CustomException(e, sys)
