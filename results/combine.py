import json
import os
import pandas as pd
import numpy as np

def log_results(results, output_path="results.json"):
    # Convert all NumPy data types to native Python types
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(i) for i in o]
        else:
            return o

    # Apply conversion and save as JSON
    cleaned_results = [convert(item) for item in results]

    with open(output_path, "w") as f:
        json.dump(cleaned_results, f, indent=2)

    print(f"[LOGS] Results saved to {output_path}")
    
dataset = "cross_domains_cross_models.csv"
df = pd.read_csv(dataset)
# Get the "source_file" column and if test.csv then put df
df = df[df["source_file"] == "test.csv"]
df = df.reset_index(drop=True)
df_label1 = df[df["label"] == 1].head(2500)
df_label0 = df[df["label"] == 0].head(2500)
df_json1 = pd.concat([df_label0, df_label1]).sample(frac=1, random_state=42).reset_index(drop=True)

print(df_json1.head())

df_label1_2 = df[df["label"] == 1].head(5000)[2500:]
df_label0_2 = df[df["label"] == 0].head(5000)[2500:]
df_json2 = pd.concat([df_label0_2, df_label1_2]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_json2.head())

results = [None] * 10000
# Look for all .json in the current directory
for filename in os.listdir("."):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
            idx = 0
            if "json1" in filename:
                df = df_json1
                results_idx = 0
            elif "json2" in filename:
                df = df_json2
                results_idx = 5000
            print(df.head())
            for data_point in data:
                data_point["text"] = df.loc[idx, "text"]
                data_point["label"] = df.loc[idx, "label"]
                results[results_idx] = data_point
                results_idx += 1
                idx += 1
            print(f"Reached {idx} for this file {filename}")

log_results(results, "raidar_overall.json")
