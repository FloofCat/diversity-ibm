import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def build_model_and_plot(X, y, feature):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'roc_curve_{feature}.png')
    
    # Print Accuracy and AUC-ROC
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC-ROC: {auc:.2f}")
    
# Read the dataset
results_json = "gpt2_overall.json"
single_features = ["entropy", "logp", "logrank", "rank"]
multi_features = ["detectllm", "diversity"]
with open(results_json, 'r') as file:
    results = json.load(file)
    
    """
    Each result in results looks like this:
    {
    {
        "entropy": ,
        "logp": ,
        "logrank": ,
        "detectllm": [
        ],
        "rank": ,
        "diversity": [
        ],
        "label": 0 or 1,
        "text": ""
    }
    """
    # Filter valid entries instead of modifying the list during iteration
    filtered_results = []
    num = 0
    
    for result in results:
        diversity = result.get("diversity")
        if isinstance(diversity, list) and len(diversity) == 11:
            filtered_results.append(result)
        else:
            num += 1
    
    print(f"Removed {num} results with invalid diversity")
    
    # Train with different features
    for feature in single_features:
        print(f"Training with feature: {feature}")
        X = [r[feature] for r in filtered_results]
        y = [r["label"] for r in filtered_results]
        X = np.array(X).reshape(-1, 1)
        Y = np.array(y)
        build_model_and_plot(X, y, feature)
        print("-------------------------------------")
    
    for feature in multi_features:
        print(f"Training with feature: {feature}")
        X = np.array([r[feature] for r in filtered_results])
        y = np.array([r["label"] for r in filtered_results])
        build_model_and_plot(X, y, feature)
        print("-------------------------------------")