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
results = [None] * len(df)
# Look for all .json in the current directory
for filename in os.listdir("."):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
            idx = filename.split("/")[-1].split("_")[-1].split("-")[0]
            idx = int(idx)
            for data_point in data:
                data_point["text"] = df.loc[idx, "text"]
                data_point["label"] = df.loc[idx, "label"]
                results[idx] = data_point
                idx += 1

log_results(results, "raidar_overall.json")
