
# Credit Card Fraud Detection System

## Project Overview

This project is a machine learning system designed to detect fraudulent credit card transactions in real time.
It addresses the severe class imbalance problem present in fraud datasets (fraud cases represent only about 0.17% of transactions) using SMOTE (Synthetic Minority Over-sampling Technique) combined with an optimized Random Forest Classifier.

The project also includes an interactive Streamlit-based Transaction Simulator that allows users to test the model using both random and injected fraud transactions.

## Key Results

- ROC-AUC Score: 0.9688
- Recall: 81%
- Precision: 81%
- False Positives: Approximately 18 misclassified normal transactions in the test set

## Folder Structure

```
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── raw/                 # Contains creditcard.csv (Kaggle dataset)
│   └── processed/           # Processed dataset with scaled features
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory data analysis and imbalance visualization
│   └── 02_Modeling.ipynb    # Model experimentation and tuning
│
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── train.py             # Training pipeline (SMOTE + Random Forest)
│   └── evaluate.py          # Evaluation and metric generation
│
├── models/
│   ├── fraud_model.pkl      # Trained Random Forest model
│   └── scaler.pkl           # Saved StandardScaler
│
├── app.py                   # Streamlit transaction simulator
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd Credit-Card-Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place the file

```
creditcard.csv
```

inside:

```
data/raw/
```

## How to Run

### 1. Data Processing

```bash
python src/data_loader.py
```

### 2. Train the Model

```bash
python src/train.py
```

This will save the trained model and scaler inside the models directory and print evaluation metrics.

### 3. Run the Simulator

```bash
streamlit run app.py
```

## Technical Approach

### Preprocessing

The dataset includes PCA-transformed features (V1–V28) which are already scaled.
The Time and Amount features are normalized using StandardScaler.

### Class Imbalance Handling

SMOTE is applied only on the training data to avoid data leakage and balance fraud and non-fraud samples.

### Model Selection

- Algorithm: Random Forest Classifier
- Estimators: 100

Random Forest was selected for its robustness, ability to handle non-linear relationships, and strong performance on tabular data.

## Performance Metrics

Confusion matrix on the test set:

- True Negatives: 56,846
- False Positives: 18
- False Negatives: 19
- True Positives: 79

## Tech Stack

- Python 3.x
- Scikit-learn
- Imbalanced-learn
- Pandas, NumPy
- Streamlit

## License

This project is licensed under the MIT License.
