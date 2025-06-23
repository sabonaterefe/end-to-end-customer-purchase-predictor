from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback

app = FastAPI()

# Load the trained model
try:
    model = joblib.load("models/random_forest_v1.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    traceback.print_exc()
    model = None

class PurchaseInput(BaseModel):
    features: list

@app.post("/predict")
def predict(data: PurchaseInput):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
        
        if not isinstance(data.features, list) or len(data.features) != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} numeric features, but received {len(data.features)}"
            )
        
        prediction = model.predict([data.features])[0]
        probability = model.predict_proba([data.features])[0][1]

        return {
            "purchase_prediction": int(prediction),
            "confidence": round(probability, 4)
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Purchase Predictor API! Use /predict to make predictions."}
