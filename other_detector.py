import torch
import numpy as np
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from baseline_models.fastdetect import FastDetect
from baseline_models.t5sentinel import T5Predictor

class OtherWorker:
    def __init__(self, model_path):
        # Initialize once per actor        
        self.fastdetect = FastDetect()
        self.t5sentinel = T5Predictor(model_path)
        
    def infer(self, text):
        return {
            "fastdetect": self.fastdetect.detect(text),
            "t5sentinel": self.t5sentinel.compute_t5(text)
        }
    
    def infer_multiple(self, texts):
        results = [None] * len(texts)
        for i, text in enumerate(tqdm(texts)):
            r = None
            try:
                r = {
                    "fastdetect": self.fastdetect.detect(text),
                    "t5sentinel": self.t5sentinel.compute_t5(text)
                }
            except:
                r = {
                    "fastdetect": 0,
                    "t5sentinel": 0
                }

                
            results[i] = r
            
        return results
    
    def shutdown(self):
        self.t5sentinel.del_models()
    
    
