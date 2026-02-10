import pickle
import pandas as pd

#import the ml model 
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# comes from MLFlow software tracks model version
MODEL_VERSION = '1.0.0'

#Get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()                              #classes - high, medium , low


def  predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    #Predict the class
    predicted_class = model.predict(df)[0]

    #Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    #Create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

    return{
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probablities": class_probs
    }

    input_df = pd.DataFrame([user_input])

    output= model.predict(input_df)[0]

    return output 