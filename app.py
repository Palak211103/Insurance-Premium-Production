from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, model, MODEL_VERSION

#import the ml model 
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# comes from MLFlow software tracks model version
MODEL_VERSION = '1.0.0'


app = FastAPI()    # app object



# human readable     
@app.get('/')
def home():
    return {'message': 'Insurance Premium Prediction API'}


# machine readable  
@app.get('/health')            #recomended tells that api is live and is working fine
def health_check():
    return{
        'status' : 'OK',
        'version':  MODEL_VERSION,
        'model_loaded': model is not None
    }

 
@app.post('/predict', response_model=PredictionResponse)      #jab output bhejenge client ke pass tab pehle vo pydantic model se validate hoga
def predict_premium(data: UserInput):  #data = input mein data milega to this function predict_premium , request body se data aega and seedha pydantic model ke pass jaega, pydantic model will work on it , find computed fields and vo palat kar data ki form men humemil jaega  
         #  1 row ka data model mein pass krna hai, input will be sent in pandas dataframe formatbecause random forest model has been trained on a dataframe model
    user_input= {
        'bmi' : data.bmi,
        'age_group' : data.age_group, 
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation

    }
    
    try:
        prediction = predict_output(user_input)

        return JSONResponse(status_code= 200, content={'response': prediction})
    
    except Exception as e:
         
         return JSONResponse(status_code=500, content=str(e))


