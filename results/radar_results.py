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

def build_model_and_plot(X, y, X_test, y_test, feature):
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
results_json = "../cross_domains_cross_models/jsons/radar_overall.json"
dataset = sys.argv[1]
single_features = ["radar"]
multi_features = []

# df = pd.read_csv(dataset)

df = pd.read_csv(dataset)
dict_df = {}
for i, row in df.iterrows():
    dict_df[row["text"]] = i

with open(results_json, 'r') as file:
    results = json.load(file)
    
    """
    Each result in results looks like this:
    {
    {
        "roberta":
    }
    """    
    # Filter between train and test sets, check if r["text"] is in the df and if the same text in the df, its in train if the column "source_file" == "train.csv" or "eval.csv", test if "test.csv"
    results_train = []
    results_test = []
    results_test_ood = []
    
    for result in results:
        # Look for column "source_file" for the index in df
        try:
            index = dict_df[result["text"]]
        except:
            continue
        source_file = df.loc[index, "source_file"]
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
    # Train with different features
    for feature in single_features:
        print(f"Training with feature: {feature}")
        X = [r[feature] for r in filtered_results]
        y = [r["label"] for r in filtered_results]
        X = np.array(X).reshape(-1, 1)
        Y = np.array(y)
        
        X_test = [r[feature] for r in results_test]
        X_test = np.array(X_test).reshape(-1, 1)
        y_test = [r["label"] for r in results_test]
        y_test = np.array(y_test)
        
        X_test_ood = [r[feature] for r in results_test_ood]
        X_test_ood = np.array(X_test_ood).reshape(-1, 1)
        y_test_ood = [r["label"] for r in results_test_ood]
        y_test_ood = np.array(y_test_ood)
        
        build_model_and_plot(X, y, X_test, y_test, feature)
        build_model_and_plot(X, y, X_test_ood, y_test_ood, feature)
        print("-------------------------------------")
    
    for feature in multi_features:
        print(f"Training with feature: {feature}")
        X = np.array([r[feature] for r in filtered_results])
        y = np.array([r["label"] for r in filtered_results])
        
        X_test = np.array([r[feature] for r in results_test])
        y_test = np.array([r["label"] for r in results_test])
        
        X_test_ood = np.array([r[feature] for r in results_test_ood])
        y_test_ood = np.array([r["label"] for r in results_test_ood])
        
        build_model_and_plot(X, y, X_test, y_test, feature)
        build_model_and_plot(X, y, X_test_ood, y_test_ood, feature)
        print("-------------------------------------")