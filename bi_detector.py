import torch
import numpy as np
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline_models.binoculars import Binoculars
from baseline_models.biscope import BiScope

class BiWorker:
    def __init__(self, model_path, model_path2):
        # Initialize once per actor
        self.binoculars_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.binoculars_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True
        )
        self.binoculars_performer = AutoModelForCausalLM.from_pretrained(
            model_path2, device_map="cuda:1", torch_dtype=torch.float16, trust_remote_code=True
        )
        
        self.binoculars = Binoculars(self.binoculars_model, self.binoculars_performer, self.binoculars_tokenizer)
        self.biscope = BiScope(self.binoculars_performer, self.binoculars_tokenizer, self.binoculars_performer, self.binoculars_tokenizer)

    def infer(self, text):
        return {
            "binoculars": self.binoculars.compute_score(text),
            "biscope": self.biscope.extract_features(text)
        }
    
    def infer_multiple(self, texts):
        results = [None] * len(texts)
        for i, text in enumerate(tqdm(texts)):
            # try:
            #     r = self.raidar.compute_raidar(text)
            # except:
            #     r = 0
                
            results[i] = {
                "binoculars": self.binoculars.compute_score(text),
                "biscope": self.biscope.extract_features(text)
            }
            
        return results
    
    def shutdown(self):
        del self.binoculars
        del self.biscope
        gc.collect()
        torch.cuda.empty_cache()