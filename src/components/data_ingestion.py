import os 
import sys
from src.Exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestionConfig:
    train_data_path: str=os.path.join('databox',"train.csv")
    test_data_path: str=os.path.join('databox',"test.csv")
    raw_data_path: str=os.path.join('databox',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered in the Data Ingestion System")
        try:
            df = pd.read_csv('Notebook\Algerian_forest_fires_cleaned_dataset.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            logging.info("Directory Created/Initialized")

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
    

        except Exception as e:
            raise CustomException(e,sys)    
        
if __name__ == "__main__":
    try:
        # Create an instance of the DataIngestion class
        data_ingestion = DataIngestion()

        # Call the initiate_data_ingestion method to perform data ingestion
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    except Exception as e :
        raise CustomException(e,sys)            