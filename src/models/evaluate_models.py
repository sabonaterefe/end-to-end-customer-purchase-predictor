#New commit
# Evaluate and compare models
import pandas as pd
import joblib
import os
import sys  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.data_processing.feature_engineer import feature_engineer

def evaluate_model(name, model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1-Score": f1_score(y_val, preds),
        "ROC-AUC": roc_auc_score(y_val, probs)
    }

    print(f"\n🔍 {name} Evaluation")
    print("-" * 40)
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))
    print("\nClassification Report:\n", classification_report(y_val, preds))
    print("✅ Score Summary:")
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")

def evaluate_all_models():
    df = pd.read_csv("data/processed/cleaned_customer_purchase_data.csv")
    df = feature_engineer(df)

    X = df.drop(columns=["PurchaseStatus"])
    y = df["PurchaseStatus"]
    _, X_val, _, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": joblib.load("models/random_forest_v1.pkl"),
        "XGBoost": joblib.load("models/xgboost_v1.joblib")
    }

    for name, model in models.items():
        evaluate_model(name, model, X_val, y_val)

if __name__ == "__main__":
    evaluate_all_models()
