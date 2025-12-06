import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Add 'src' to path so we can import data_loader
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data

def train_model():
    # 1. Load Processed Data (NEW PATH)
    data_path = "data/processed/creditcard_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Run 'python src/data_loader.py' first!")
        return

    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Split Features & Target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Original Training Shape: {X_train.shape}")
    print(f"Fraud cases in Train: {sum(y_train)}")

    # 4. Apply SMOTE (The Magic Step) 🪄
    print("\nApplying SMOTE to generate synthetic fraud cases...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"Resampled Training Shape: {X_train_res.shape}")
    print(f"Fraud cases after SMOTE: {sum(y_train_res)}")
    print("(Now we have 50/50 balance!)")

    # 5. Train Model (Random Forest)
    print("\nTraining Random Forest (this might take a minute)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability for ROC-AUC

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # 7. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    print("\nModel saved to 'models/fraud_model.pkl'")

if __name__ == "__main__":
    train_model()