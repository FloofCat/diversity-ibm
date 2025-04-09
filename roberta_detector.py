import torch
import numpy as np
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from baseline_models.roberta import RobertaBase

class RobertaWorker:
    def __init__(self, model_path):
        # Initialize once per actor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True
        )
        
        self.roberta = RobertaBase(self.model, self.tokenizer)
        
    def infer(self, text):
        return {
            "roberta": self.roberta.predict(text)
        }
    
    def infer_multiple(self, texts):
        results = [None] * len(texts)
        for i, text in enumerate(tqdm(texts)):
            r = None
            try:
                r = self.roberta.predict(text)
            except:
                r = 0
                
            results[i] = {
                "roberta": r
            }
            
        return results
    
    def shutdown(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
    
    
