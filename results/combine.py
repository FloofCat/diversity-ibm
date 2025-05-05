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
    
dataset = "./model-cache/raid/train.csv"
df = pd.read_csv(dataset)

results = [None] * len(df)
# Look for all .json in the current directory
for filename in os.listdir("."):
    if filename.endswith(".json") and filename.startswith("test-fastdetect"):
        # Get file idx.
        idx = int(filename.split("-")[-1].split(".")[0])
        idx = idx - 3
        idx *= 1000000
        print(f"Processing file {filename} with idx {idx}")
        with open(filename, "r") as f:
            data = json.load(f)
            print(df.head())
            for data_point in data:
                addition = {}
                addition["text"] = df.loc[idx, "generation"]
                addition["label"] = df.loc[idx, "model"]
                addition["fastdetect"] = data_point["diversity"]
                results[idx] = addition
                idx += 1
            print(f"Reached {idx} for this file {filename}")

log_results(results, "fastdetect_overall.json")
