#xgboost
# Train an XGBoost model for customer purchase prediction
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import os
import sys  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.data_processing.feature_engineer import feature_engineer

def train_xgboost():
    # Load and preprocess data
    df = pd.read_csv("data/processed/cleaned_customer_purchase_data.csv")
    df = feature_engineer(df)

    X = df.drop(columns=["PurchaseStatus"])
    y = df["PurchaseStatus"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate quick baseline performance
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"✅ XGBoost Validation Accuracy: {acc:.4f}")

    # Save model
    dump(model, "models/xgboost_v1.joblib")
    print("📦 Model saved as models/xgboost_v1.joblib")

if __name__ == "__main__":
    train_xgboost()
