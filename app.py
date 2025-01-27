from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Temperature=request.form.get('Temperature'),
            RH=request.form.get('RH'),
            Ws=request.form.get('Ws'),
            Rain=float(request.form.get('Rain')),
            FFMC=float(request.form.get('FFMC')),
            DMC=float(request.form.get('DMC')),
            DC=float(request.form.get('DC')),
            ISI=float(request.form.get('ISI')),
            BUI=float(request.form.get('BUI')),
            Classes=request.form.get('Classes'),
            Region=request.form.get('Region')
        )

        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()

        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)