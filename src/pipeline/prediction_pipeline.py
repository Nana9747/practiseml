import sys 
import os
import pandas as pd 
from src.Exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("databox","model.pkl")
            preprocessor_path = os.path.join("databox","preprocessor.pkl")

            model =load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds


        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, Temperature, RH, Ws, Rain, FFMC, 
                 DMC, DC, ISI, BUI,Classes, Region):

        self.Temperature = Temperature
        self.RH = RH
        self.Ws = Ws
        self.Rain = Rain
        self.FFMC = FFMC
        self.DMC = DMC
        self.DC = DC
        self.ISI = ISI
        self.BUI = BUI
        self.Classes = Classes
        self.Region = Region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Temperature": [self.Temperature],
                "RH":[self.RH],
                "Ws":[self.Ws],
                "Rain":[self.Rain],
                "FFMC":[self.FFMC],
                "DMC":[self.DMC],
                "DC":[self.DC],
                "ISI":[self.ISI],
                "BUI":[self.BUI],
                "Classes":[self.Classes],
                "Region":[self.Region]
            }

            return pd.DataFrame(custom_data_input_dict)        

        except Exception as e:
            raise CustomException(e,sys)    