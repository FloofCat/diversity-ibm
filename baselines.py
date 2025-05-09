import torch
import gc
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import multiprocessing as mp
from baseline_models.downloader import Downloader
from gpt2_detector import GPT2Worker
from roberta_detector import RobertaWorker
from other_detector import run

lim1 = 0
lim2 = 25000
INDEX = 1

class Baselines:
    def __init__(self):
        # Download models if not already downloaded
        self.cache_dir = "./model-cache"
        self.models = {
            "gpt2": "gpt2",
            "roberta": "openai-community/roberta-base-openai-detector",
            "radar": "TrustSafeAI/RADAR-Vicuna-7B",
            "binoculars_observer": "tiiuae/falcon-7b",
            "binoculars_performer": "tiiuae/falcon-7b-instruct",
            "raidar": "tiiuae/falcon-7b-instruct"
        }
        self.types = {
            "gpt2": AutoModelForCausalLM,
            "roberta": AutoModelForSequenceClassification,
            "radar": AutoModelForSequenceClassification,
            "binoculars_observer": AutoModelForCausalLM,
            "binoculars_performer": AutoModelForCausalLM,
            "raidar": AutoModelForCausalLM
        }

        self.downloader = Downloader(self.models, self.types, self.cache_dir)

    def log_results(self, results, output_path="results.json"):
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

    def detect_radar(self, texts):
        self.radar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/radar", use_fast=False, trust_remote_code=True)
        self.radar_model = self.types["radar"].from_pretrained(f"{self.cache_dir}/radar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] RADAR Loaded")

        results = [None] * len(texts)

        self.radar = RADAR(self.radar_model, self.radar_tokenizer)

        for i, text in enumerate(tqdm(texts)):
            results[i] = {
                "radar": self.radar.detect_probability(text)
            }

        del self.radar_tokenizer
        del self.radar_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] RADAR Completed")

        return results

    def detect_binoculars_biscope(self, texts):
        self.binoculars_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/binoculars_observer", use_fast=False, trust_remote_code=True)
        self.binoculars_observer_model = self.types["binoculars_observer"].from_pretrained(f"{self.cache_dir}/binoculars_observer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        self.binoculars_performer_model = self.types["binoculars_performer"].from_pretrained(f"{self.cache_dir}/binoculars_performer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] Binoculars Loaded")

        self.binoculars = Binoculars(self.binoculars_observer_model, self.binoculars_performer_model, self.binoculars_tokenizer)
        self.biscope = BiScope(self.binoculars_performer_model, self.binoculars_tokenizer, self.binoculars_performer_model, self.binoculars_tokenizer)

        results = [None] * len(texts)

        for i, text in enumerate(tqdm(texts)):
            results[i] = {
                "binoculars": self.binoculars.compute_score(text),
                "biscope": self.biscope.extract_features(text)
            }

        del self.binoculars_tokenizer
        del self.binoculars_performer_model
        del self.binoculars_observer_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] Binoculars Completed")

        return results
    
    def detect_raidar(self, texts):
        self.raidar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/raidar", use_fast=False, trust_remote_code=True)
        self.raidar_model = self.types["raidar"].from_pretrained(f"{self.cache_dir}/raidar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] RAIDAR Loaded")

        self.raidar = RAIDAR(self.raidar_model, self.raidar_tokenizer)

        results = [None] * len(texts)

        for i, text in enumerate(tqdm(texts)):
            results[i] = {
                "raidar": self.raidar.compute_raidar(text)
            }

        del self.raidar_tokenizer
        del self.raidar_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] RAIDAR Completed")

        return results

    def detect_others(self, texts):
        self.fastdetect = FastDetect()
        self.t5sentinel = T5Predictor("./model-cache/T5Sentinel.0613.pt")
        print("[LOGS] Others Loaded")

        results = [None] * len(texts)

        for i, text in enumerate(tqdm(texts)):
            results[i] = {
                "fastdetect": self.fastdetect.detect(text),
                "t5sentinel": self.t5sentinel.compute_t5(text)
            }

        self.t5sentinel.del_models()
        print("[LOGS] Others completed")

        return results

if __name__ == "__main__":
    print("What is happening? Has this docker image actually imported yet?")
    baselines = Baselines()
    gpt2_worker = GPT2Worker("./model-cache/binoculars_performer")
    # roberta_worker = RobertaWorker("./model-cache/roberta")
    # other_worker = OtherWorker("meow")
    
    train_df = pd.read_csv("./model-cache/raid/train.csv")
    texts = train_df["generation"][(INDEX-1)*1000000:INDEX*1000000].tolist()

    # run(texts) # other worker
    baselines.log_results(gpt2_worker.infer_multiple(texts), f"train-gpt2_results_{INDEX}.json")
