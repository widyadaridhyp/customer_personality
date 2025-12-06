from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.model_pipeline import ModelPipeline

app = FastAPI()

pipeline = ModelPipeline()
pipeline.load()

class Payload(BaseModel):
    data: list

@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame(payload.data)

    pred, prob = pipeline.predict(df)

    return {
        "prediction": pred.tolist(),
        "probability": prob.tolist()
    }
