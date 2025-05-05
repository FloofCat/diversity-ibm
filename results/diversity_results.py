import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import sys
from raid import run_detection, run_evaluation
from raid.utils import load_data
from gpt2_detector import GPT2Worker
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm


PRIMARY_DATASET = "./model-cache/raid/train.csv"
SECONDARY_DATASET = "./model-cache/raid/test.csv"
diversity_model = GPT2Worker("./model-cache/gpt2")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def build_model_a(X, y):
    """Build and train the fastdetect model (Model A)"""
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        scale_pos_weight=(len(y) - sum(y)) / sum(y)
    )
    model.fit(X, y)
    return model

def build_model_b(X, y):
    """Build and train the diversity model (Model B)"""
    print("Building Model B...")
    print("Shape of X:", X.shape)
    
    model = xgb.XGBClassifier(random_state=42, 
        scale_pos_weight=(len(y) - sum(y)) / sum(y),
        max_depth=12,
        n_estimators=200,
        colsample_bytree=0.8,
        subsample=0.7,
        min_child_weight=5,
        gamma=1.0
    )
    model.fit(X, y)
    
    return model

def evaluate_and_plot(y_test, y_pred, y_prob, model_name):
    """Evaluate and plot results for a given model"""
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, y_prob[:, 1])
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.6f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name}.png')
    
    # Print results
    print(f"\n----- {model_name} Results -----")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"AUC-ROC: {auc:.6f}")
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Accuracy per label
    labels = np.unique(y_test)
    for label in labels:
        idx = y_test == label
        label_acc = np.mean(y_pred[idx] == y_test[idx])
        print(f"Accuracy for label {label}: {label_acc:.6f}")

# Read the dataset
results_json = "diversity_overall.json"
results2_json = "fastdetect_overall.json"
df = pd.read_csv(PRIMARY_DATASET)

results_test_json = "test_raid_gpt2_overall.json"
results2_test_json = "test_raid_fastdetect_overall.json"
df2 = pd.read_csv(SECONDARY_DATASET)

# Create dictionary for fast lookup
dict_df = {}
for i, row in df.iterrows():
    dict_df[row["generation"]] = i

# Load fastdetect features
dict_fastdetect = {}
with open(results2_json, "r") as f:
    results = json.load(f)
    for result in results:
        dict_fastdetect[result["text"]] = result["fastdetect"]

# Load diversity features and process data
with open(results_json, 'r') as file:

    results = json.load(file)
    # Filter valid entries
    filtered_results = []
    num_invalid = 0
    
    for result in results:
        if result is None:
            num_invalid += 1
            continue
            
        diversity = result.get("diversity")
        if isinstance(diversity, list) and len(diversity) == 11:
            filtered_results.append(result)
        else:
            num_invalid += 1
    
    print(f"Removed {num_invalid} results with invalid diversity")

    train_diversity = []
    train_fastdetect = []
    train_combined = []
    y_train = []
    for result in filtered_results:
        try:
            index = dict_df[result["text"]]
        except:
            continue
        # source_file = df.loc[index, "source_file"]
        
        # Append fastdetect feature to diversity features
        fastdetect_score = dict_fastdetect.get(result["text"])
        if fastdetect_score is None:
            continue
            
        # Create a copy of the result with both features
        train_diversity.append(result["diversity"].copy())
        train_fastdetect.append([fastdetect_score])
        bleh = result["diversity"].copy()
        bleh.append(fastdetect_score)
        train_combined.append(bleh)
        
        y_train.append(result["label"] != "human")
    
    X_a_train = np.array(train_fastdetect)
    X_b_train = np.array(train_diversity)
    X_combined_train = np.array(train_combined)
    y_train = np.array(y_train)
    
    print("\n----- Training Models -----")
    # Train individual models
    model_diversity = build_model_b(X_b_train, y_train)
    model_fastdetect = build_model_a(X_a_train, y_train)
    combined_model =  xgb.XGBClassifier(
        random_state=42, 
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        max_depth=12,
        n_estimators=200,
        colsample_bytree=0.8,
        subsample=0.7,
        min_child_weight=5,
        gamma=1.0
    )
        
    combined_model.fit(X_combined_train, y_train)
   
    # ===== Evaluate Ensemble Model =====
    print("\nGetting evaluations on testbed")
    dict_test_div = {}
    with open(results_test_json, 'r') as f:
        data = json.load(f)
        for item in data:
            dict_test_div[item['text']] = item['diversity']
    
    dict_test_fastdetect = {}
    with open(results2_test_json, 'r') as f:
        data = json.load(f)
        for item in data:
            dict_test_fastdetect[item['text']] = item['fastdetect']
    
    def my_combdetector(texts: list[str]) -> list[float]:
        error = 0
        predictions = []
        for text in tqdm(texts):
            X_div = dict_test_div[text]
            if X_div is None:
                print("Missing diversity features for text:", text)
                predictions.append(0.0)
                error += 1
                continue
            try:
                X_fast = dict_test_fastdetect[text]
            except:
                raise ValueError(f"Missing fastdetect features for text: {text}")
            X_combine = X_div.copy()
            X_combine.append(X_fast)
            predictions.append(combined_model.predict_proba([X_combine])[0, 1])
        print("Error count:", error)
        return predictions
    
    def my_detector(texts: list[str]) -> list[float]:
        predictions = []
        error2 = 0
        for text in texts:
            X_div = dict_test_div[text]
            if X_div is None:
                print("Missing diversity features for text:", text)
                predictions.append(0.0)
                error2 += 1
                continue
            else:
                predictions.append(model_diversity.predict_proba([X_div])[0, 1])
        print("Error count:", error2)
        return predictions
    
    # Run your detector on the dataset
    predictions = run_detection(my_combdetector, df2)
    
    with open('predictions-combined.json', 'w') as f:
        json.dump(predictions, f)
        
    predictions = run_detection(my_detector, df2)
    with open('predictions-diversity.json', 'w') as f:
        json.dump(predictions, f)
            
    
    print("\n-------------------------------------")
    print("Ensemble modeling complete!")