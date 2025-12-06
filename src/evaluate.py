import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

def plot_confusion_matrix(y_test, y_pred, save_path="reports/figures/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (SMOTE)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")

def plot_roc_curve(y_test, y_prob, save_path="reports/figures/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print(f"ROC Curve saved to {save_path}")
    