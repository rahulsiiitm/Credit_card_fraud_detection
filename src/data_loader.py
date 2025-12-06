import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_process_data(raw_filepath, processed_filepath):
    """
    Loads raw data, scales Time/Amount, and saves the processed file.
    """
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"File not found: {raw_filepath}")
        
    print(f"Loading raw data from {raw_filepath}...")
    df = pd.read_csv(raw_filepath)
    
    # 1. Scale 'Time' and 'Amount'
    print("Scaling 'Time' and 'Amount' columns...")
    scaler = StandardScaler()
    
    # We use fit_transform to learn the mean/std and apply it
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original raw columns (so the model only sees scaled versions)
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # 2. Save the Scaler (Important for app.py!)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to 'models/scaler.pkl'")
    
    # 3. Save the Processed CSV
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to {processed_filepath}")
    
    return df

if __name__ == "__main__":
    # Define paths
    raw_path = os.path.join('data', 'raw', 'creditcard.csv')
    proc_path = os.path.join('data', 'processed', 'creditcard_processed.csv')
    
    load_and_process_data(raw_path, proc_path)