from fastapi import FastAPI, HTTPException
from src.api.schemas import FeaturesInput
import joblib
import traceback

app = FastAPI(
    title="🛍️ Customer Purchase Predictor API",
    description="Predicts the likelihood of a purchase based on customer features.",
    version="1.0.0",
)

# Load the trained model
try:
    model = joblib.load("models/random_forest_v1.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    traceback.print_exc()
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Purchase Predictor API! Use /predict to make predictions."}

@app.post("/predict")
def predict(data: FeaturesInput):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
        
        features = data.features

        if len(features) != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} features, but received {len(features)}"
            )
        
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0][1]

        return {
            "purchase_prediction": int(prediction),
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")
