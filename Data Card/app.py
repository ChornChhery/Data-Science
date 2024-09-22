from fastapi import FastAPI
from pydantic import BaseModel
import joblib
class InputData(BaseModel):
    input_data1 : float
    input_data2 : float

model = joblib.load("LinearModel.pkl")

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_values = [[data.input_data1, data.input_data2]]

    prediction = model.predict(input_values)[0]
    return {"prediction": prediction}
    