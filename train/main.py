import numpy as np
import pandas as pd
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from raid import run_detection, run_evaluation
from raid.utils import load_data

from diversity import SurprisalModel
from radar import RADAR
from roberta import RobertaBase
from fastdetect import FastDetect

# RAID
train_df = load_data(split="train")

surprisal_model = SurprisalModel()
radar_model = RADAR()
roberta_model = RobertaBase()
fastdetect = FastDetect()

def extract_features(texts):
    

# -------------------------------
# Train XGBoost Classifier
# -------------------------------
print("Training XGBoost Classifier...")
clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    colsample_bytree=0.8,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Print feature importances
feature_importances = clf.feature_importances_
print("\nFeature Importances:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance:.4f}")
    
print("Training RF + XGBoost Classifier...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

rf = RandomForestClassifier(n_estimators=300, max_depth=7)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=7)
clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

clf.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Train XGB on FastDetect
print("Training XGBoost Classifier on FastDetect...")
clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    colsample_bytree=0.8,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)

clf.fit(X_train_fastdetect.reshape(-1, 1), y_train)

# -------------------------------
# 
# -------------------------------
y_pred = clf.predict(X_test_fastdetect.reshape(-1, 1))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print feature importances
feature_importances = clf.feature_importances_
print("\nFeature Importances:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i+1}: {importance:.4f}")


