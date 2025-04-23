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

PRIMARY_DATASET = sys.argv[1]

def mask_fastdetect(X, y, replacement="zero"):
    X_mod = X.copy()
    X_mod[y == 1, -1] = 0
    return X_mod

def build_model_and_plot(X, y, X_test, y_test, feature):
    
    # X = mask_fastdetect(X, y, replacement="zero")
    # X_test = mask_fastdetect(X_test, y, replacement="zero")
    # Unbalanced, train balanced, RF 
    model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42, scale_pos_weight=(len(y) - sum(y)) / sum(y))
    model.fit(X, y)
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
    
    # Save the model
    model.save_model("div_model.json")
    
    # Print Accuracy and AUC-ROC
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC-ROC: {auc:.2f}")
    
    # Print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Accuracy per label
    labels = np.unique(y_test)
    for label in labels:
        idx = y_test == label
        label_acc = np.mean(y_pred[idx] == y_test[idx])
        print(f"Accuracy for label {label}: {label_acc}")
    
# Read the dataset
results_json = "../cross_domains_cross_models/jsons/diversity_overall.json"
results2_json = "../cross_domains_cross_models/jsons/fastdetect_overall.json"


df = pd.read_csv(PRIMARY_DATASET)
dict_df = {}
for i, row in df.iterrows():
    dict_df[row["text"]] = i

dict_fastdetect = {}
with open(results2_json, "r") as f:
    results = json.load(f)
    for result in results:
        dict_fastdetect[result["text"]] = result["fastdetect"]

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
    
    # Filter between train and test sets, check if r["text"] is in the df and if the same text in the df, its in train if the column "source_file" == "train.csv" or "eval.csv", test if "test.csv"
    results_train = []
    results_test = []
    results_test_ood = []
    
    for result in filtered_results:
        # Look for column "source_file" for the index in df
        try:
            index = dict_df[result["text"]]
        except:
            continue
        source_file = df.loc[index, "source_file"]
        
        # Appending fastdetect here
        fastdetect_score = dict_fastdetect[result["text"]]
        
        result["diversity"].append(fastdetect_score)
        # Add transformed versions of the fastdetect score to diversity
        result["diversity"].append(fastdetect_score ** 2)
        result["diversity"].append(np.log(1+fastdetect_score))
        
        if source_file == "train.csv" or source_file == "valid.csv":
            results_train.append(result)
        elif source_file == "test.csv":
            results_test.append(result)
        elif source_file == "test_ood.csv":
            results_test_ood.append(result)
    
    
    # Check number of samples with label 0 and 1 in train and test sets
    num_samples_label_0_train = sum([r["label"] == 0 for r in results_train])
    num_samples_label_1_train = sum([r["label"] == 1 for r in results_train])
    num_samples_label_0_test = sum([r["label"] == 0 for r in results_test])
    num_samples_label_1_test = sum([r["label"] == 1 for r in results_test])
    print(f"Number of samples with label 0 in train set: {num_samples_label_0_train}")
    print(f"Number of samples with label 1 in train set: {num_samples_label_1_train}")
    print(f"Number of samples with label 0 in test set: {num_samples_label_0_test}")
    print(f"Number of samples with label 1 in test set: {num_samples_label_1_test}")
    num_samples_label_0_test_ood = sum([r["label"] == 0 for r in results_test_ood])
    num_samples_label_1_test_ood = sum([r["label"] == 1 for r in results_test_ood])
    print(f"Number of samples with label 0 in test_ood set: {num_samples_label_0_test_ood}")
    print(f"Number of samples with label 1 in test_ood set: {num_samples_label_1_test_ood}")

    filtered_results = results_train
    feature = "diversity"
    X = np.array([r[feature] for r in filtered_results])
    y = np.array([r["label"] for r in filtered_results])
    
    X_test = np.array([r[feature] for r in results_test])
    y_test = np.array([r["label"] for r in results_test])
    
    X_test_ood = np.array([r[feature] for r in results_test_ood])
    y_test_ood = np.array([r["label"] for r in results_test_ood])
    
    build_model_and_plot(X, y, X_test, y_test, feature)
    build_model_and_plot(X, y, X_test_ood, y_test_ood, feature)
    print("-------------------------------------")