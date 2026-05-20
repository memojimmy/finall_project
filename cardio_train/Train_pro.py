from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json

app = FastAPI()

# load model
model = joblib.load("heart_model.pkl")

# load features
with open("features.json", "r") as f:
    features = json.load(f)


# request schema
class HeartData(BaseModel):

    age: float
    sex: float
    chest: float
    resting_blood_pressure: float
    serum_cholestoral: float
    fasting_blood_sugar: float
    resting_electrocardiographic_results: float
    maximum_heart_rate_achieved: float
    exercise_induced_angina: float
    oldpeak: float
    slope: float
    number_of_major_vessels: float
    thal: float


@app.get("/")
def root():
    return {
        "message": "Heart Disease API Running"
    }


@app.post("/predict")
def predict(data: HeartData):

    input_data = np.array([[
        data.age,
        data.sex,
        data.chest,
        data.resting_blood_pressure,
        data.serum_cholestoral,
        data.fasting_blood_sugar,
        data.resting_electrocardiographic_results,
        data.maximum_heart_rate_achieved,
        data.exercise_induced_angina,
        data.oldpeak,
        data.slope,
        data.number_of_major_vessels,
        data.thal
    ]])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0]),
        "status": (
            "Heart Disease Detected"
            if prediction[0] == 1
            else "Healthy"
        )
    }