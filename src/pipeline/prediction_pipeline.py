import sys
import os
from src.logger import logging
from src.execption import CustomException
import pandas as pd

from src.utils import load_object



class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            # Getting paths for preprocessor and model pickled file
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Loading data from pickle file to objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Scaling using preprocessor object
            data_scaled = preprocessor.transform(features)
            # Predicting using model object
            pred = model.predict(data_scaled)

            return pred


        except Exception as e:
            logging.info("Exception has occured in prediction")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str) -> None:
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df
        
        except Exception as e:
            logging.info("Exception occured at gathering data")
            raise CustomException(e,sys)

        
