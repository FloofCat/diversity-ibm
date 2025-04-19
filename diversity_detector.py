import torch
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from baseline_models.diversity import Diversity
# from baseline_models.fastdetect import FastDetect
from tqdm import tqdm

class DiversityWorker:
    def __init__(self, model_path):
        # Initialize once per actor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True
        )

        # Init detection components
        # self.fastdetect = FastDetect()
        self.diversity = Diversity(self.model, self.tokenizer)
    
    def infer(self, text):
        return {
            "diversity": self.diversity.compute_features(text)
        }
        
    def infer_multiple(self, texts):
        results = [None] * len(texts)
    
        for i, text in enumerate(tqdm(texts)):
            try:
                diversity = self.diversity.compute_features(text)
            except:
                diversity = 0
    
            results[i] = {
                "diversity": diversity
            }
    
        return results

    def shutdown(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()