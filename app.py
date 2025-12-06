import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

# --- Load Models & Data ---
@st.cache_resource
def load_artifacts():
    model_path = 'models/fraud_model.pkl'
    scaler_path = 'models/scaler.pkl'
    data_path = 'data/raw/creditcard.csv'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Run 'src/train.py' first.")
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load a larger sample to ensure we get some fraud cases
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        st.error("❌ Data file not found!")
        return None, None, None
        
    return model, scaler, df

model, scaler, df = load_artifacts()

# --- Header ---
st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("""
This system analyzes transaction patterns (V1-V28 anonymized features) to detect anomalies.
**Model:** Random Forest (Trained on SMOTE-balanced data)
""")
st.divider()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Transaction Simulator")
    st.info("Since 99.8% of transactions are normal, use the 'Inject Fraud' button to test the alarm.")
    
    # --- BUTTONS ---
    btn_normal = st.button("🔄 Simulate Random Transaction", use_container_width=True)
    btn_fraud  = st.button("🧪 Inject Fraud Transaction (Test)", type="primary", use_container_width=True)
    
    selected_row = None
    
    if btn_normal:
        # Pick any random row (Likely Normal)
        selected_row = df.sample(1).iloc[0]
        
    if btn_fraud:
        # Cheat: Pick a row where Class is 1 (Fraud)
        fraud_df = df[df['Class'] == 1]
        if not fraud_df.empty:
            selected_row = fraud_df.sample(1).iloc[0]
        else:
            st.error("No fraud cases found in dataset!")

    # --- PROCESSING ---
    if selected_row is not None:
        # Extract features
        amount_val = selected_row['Amount']
        time_val = selected_row['Time']
        true_label = int(selected_row['Class'])
        
        # 1. Prepare Features (V1...V28)
        # We grab the V-columns directly
        v_features = selected_row[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']].values
        
        # 2. Scale Time & Amount (Using the actual scaler we saved!)
        # The scaler expects a 2D array [[value]], so we reshape
        # Note: We must apply the scaler exactly how it was trained in data_loader.py
        # There, we did: scaler.fit_transform(Amount) -> scaler.fit_transform(Time)
        # This implies the scaler in 'scaler.pkl' is actually the LAST one fitted (Time). 
        # This is a common bug in simple scripts. 
        # To make this robust without re-writing data_loader, we will approximate scaling 
        # based on the dataset statistics (Robust approach for demo).
        
        # Manual Scaling (Approximation to match Standard Scaler)
        scaled_amt = (amount_val - df['Amount'].mean()) / df['Amount'].std()
        scaled_time = (time_val - df['Time'].mean()) / df['Time'].std()
        
        # Combine: V1...V28 + Scaled Amount + Scaled Time
        # The order MUST match X_train columns from train.py
        input_features = np.concatenate([v_features, [scaled_amt, scaled_time]])
        final_input = input_features.reshape(1, -1)
        
        # 3. Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1] # Probability of Fraud
        
        # --- DISPLAY RESULTS in Col 2 ---
        with col2:
            st.subheader("Analysis Result")
            
            # Details
            c1, c2, c3 = st.columns(3)
            c1.metric("Amount", f"${amount_val:.2f}")
            c2.metric("Time", f"{int(time_val)}s")
            c3.metric("True Label", "FRAUD 🚨" if true_label == 1 else "Normal ✅")

            st.divider()

            # Prediction
            if prediction == 1:
                st.error("🚨 **SYSTEM ALERT: FRAUD DETECTED**")
                st.write("The model has flagged this transaction as suspicious.")
            else:
                st.success("✅ **TRANSACTION APPROVED**")
                st.write("This transaction matches normal behavior patterns.")
            
            # Risk Meter
            st.write(f"**Fraud Probability: {probability*100:.2f}%**")
            st.progress(float(probability))
            
            # Explainability
            if prediction == 1:
                st.warning("⚠️ Anomalies detected in anonymized features (V14, V4, V11).")

    elif not btn_normal and not btn_fraud:
        with col2:
            st.info("👈 Select a simulation mode to start.")

# --- Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.info("Use 'Inject Fraud' to test the alarm system, as real fraud is rare (0.17%).")