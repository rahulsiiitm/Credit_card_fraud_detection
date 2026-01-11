import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "visualizations")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

def evaluate_model():
    # 1. Check if Data and Model exist
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # 3. Preprocess / Split
    # We split first to avoid data leakage during scaling
    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Splitting data (test_size=0.2, random_state=42)...")
    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Feature Scaling (Fixing the ValueError)
    print("Scaling 'Amount' and 'Time' features...")
    
    # Initialize scaler
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    # Fit on training data AND create new columns
    X_train['scaled_amount'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1, 1))
    X_train['scaled_time'] = scaler_time.fit_transform(X_train['Time'].values.reshape(-1, 1))
    
    # Transform test data using the fitted scalers
    X_test['scaled_amount'] = scaler_amount.transform(X_test['Amount'].values.reshape(-1, 1))
    X_test['scaled_time'] = scaler_time.transform(X_test['Time'].values.reshape(-1, 1))

    # Drop the original 'Amount' and 'Time' columns as the model doesn't expect them
    X_test = X_test.drop(['Amount', 'Time'], axis=1)
    
    # Note: X_train is not used for prediction, but we processed it to fit the scaler correctly.

    # 5. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 6. Make Predictions
    print("Predicting on test set...")
    try:
        y_pred = model.predict(X_test)
        
        # Try getting probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
    except ValueError as e:
        print(f"\nCRITICAL ERROR during prediction: {e}")
        print("Tip: Ensure the columns in X_test match exactly what the model was trained on.")
        print(f"Current columns in X_test: {list(X_test.columns)}")
        return

    # 7. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*30)
    print("MODEL PERFORMANCE METRICS")
    print("="*30)
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC-AUC Score: {roc:.4f}")
    print("="*30)

    # 8. Generate and Save Confusion Matrix
    print("\nGenerating Confusion Matrix Plot...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Create directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Credit Card Fraud Detection')
    
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Confusion Matrix saved to: {OUTPUT_IMAGE_PATH}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()