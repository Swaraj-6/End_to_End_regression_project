import os
import sys
from src.logger import logging
from src.execption import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split


from src.components.model_training import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformatio import Data_transformation

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def model_train(self):
        obj=DataIngestion()
        train_data_path, test_data_path, flag=obj.initiate_data_ingestion()
        data_transformation = Data_transformation()
        train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        best_model, best_score =  model_trainer.initate_model_training(train_arr,test_arr)

        return best_model, best_score, flag



# if __name__=='__main__':
#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = Data_transformation()
#     train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
#     model_trainer=ModelTrainer()
#     model_trainer.initate_model_training(train_arr,test_arr)


