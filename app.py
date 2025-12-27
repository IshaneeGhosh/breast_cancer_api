from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Breast Cancer Tumor Classification API")

class CancerInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "Breast Cancer API is running"}

@app.post("/predict")
def predict(data: CancerInput):
    input_array = np.array(data.features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    result = "Malignant" if prediction == 1 else "Benign"
    return {"prediction": result}
