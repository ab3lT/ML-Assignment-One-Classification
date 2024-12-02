from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your trained model
model = joblib.load('model.pkl')  # Replace with your actual model path

# Create the FastAPI app instance
app = FastAPI()

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    features: list

# Define a route to test if the server is running
@app.get("/")
def read_root():
    return {"message": "Model Deployment with FastAPI!"}

# Define the predict route to handle POST requests
@app.post("/predict")
def predict(request: PredictionRequest):
    features = request.features  # Get features from the request
    prediction = model.predict([features])  # Replace with your actual model prediction logic
    return {"prediction": prediction.tolist()}  # Return prediction as a list
