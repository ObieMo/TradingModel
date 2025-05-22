from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("model_lstm.h5")

app = FastAPI()

class PriceInput(BaseModel):
    prices: list  # Expecting list of 100 floats

def predict_next_50(prices):
    # Preprocess the input (reshape, normalize, etc.)
    # Assumes prices is a list of 100 floats
    input_array = np.array(prices).reshape(1, 100, 1)
    prediction = model.predict(input_array)
    return prediction[0].tolist()

@app.post("/predict")
async def predict_prices(input_data: PriceInput):
    if len(input_data.prices) != 100:
        raise HTTPException(status_code=400, detail="Exactly 100 prices required")
    try:
        result = predict_next_50(input_data.prices)
        return {"predicted_prices": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
