import pandas as pd
import joblib
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from typing import Optional, Any

# Ensure src is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.feature_engineer import feature_engineer

def evaluate_model(model: Optional[Any] = None, 
                   X_val: Optional[pd.DataFrame] = None, 
                   y_val: Optional[pd.Series] = None) -> None:
    """
    Evaluate a classification model on validation data.

    Supports both binary and multiclass targets. Loads default model 
    and data if inputs are not provided.

    Parameters
    ----------
    model : Optional[Any]
        The trained classification model to evaluate. If None, a default
        model will be loaded.
    X_val : Optional[pd.DataFrame]
        The validation features. If None, default validation data will
        be loaded.
    y_val : Optional[pd.Series]
        The true labels for the validation data. If None, default labels
        will be loaded.
    """
    if model is None or X_val is None or y_val is None:
        df = pd.read_csv("data/processed/cleaned_customer_purchase_data.csv")
        df = feature_engineer(df)

        X = df.drop(columns=["PurchaseStatus"])
        y = df["PurchaseStatus"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        try:
            model = joblib.load("models/random_forest_v1.pkl")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return

    preds = model.predict(X_val)
    num_classes = len(np.unique(y_val))
    average = "binary" if num_classes == 2 else "weighted"

    # Evaluation metrics
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, average=average)
    recall = recall_score(y_val, preds, average=average)
    f1 = f1_score(y_val, preds, average=average)

    try:
        roc_auc = roc_auc_score(y_val, model.predict_proba(X_val), multi_class="ovr" if num_classes > 2 else "raise")
    except Exception:
        roc_auc = "N/A"

    # Display results
    print("📊 Evaluation Results:\n")
    print("📊 Confusion Matrix:\n", confusion_matrix(y_val, preds))
    print("\n📈 Classification Report:\n", classification_report(y_val, preds))
    print("✅ Metrics Summary:")
    print(f"🎯 Accuracy  : {accuracy:.4f}")
    print(f"🎯 Precision : {precision:.4f}")
    print(f"🎯 Recall    : {recall:.4f}")
    print(f"🎯 F1-Score  : {f1:.4f}")
    print(f"🔍 ROC-AUC   : {roc_auc}")

if __name__ == "__main__":
    evaluate_model()