import pandas as pd
import joblib
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
from src.data_processing.feature_engineer import feature_engineer

def evaluate_model():
    # Load processed data
    df = pd.read_csv("data/processed/cleaned_customer_purchase_data.csv")
    df = feature_engineer(df)

    X = df.drop(columns=["PurchaseStatus"])
    y = df["PurchaseStatus"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load("models/random_forest_v1.pkl")
    preds = model.predict(X_val)

    # Evaluation metrics
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    print("📊 Evaluation Results:\n\n" )

    print("📊 Confusion Matrix:\n", confusion_matrix(y_val, preds))
    print("\n📈 Classification Report:\n", classification_report(y_val, preds))
    print("✅ Metrics Summary:")
    print(f"🎯 Accuracy  : {accuracy:.4f}")
    print(f"🎯 Precision : {precision:.4f}")
    print(f"🎯 Recall    : {recall:.4f}")
    print(f"🎯 F1-Score  : {f1:.4f}")
    print(f"🔍 ROC-AUC   : {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate_model()
