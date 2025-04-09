import torch
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from baseline_models.diversity import Diversity
from baseline_models.entropy import Entropy
from baseline_models.logp import LogP
from baseline_models.logrank import LogRank
from baseline_models.lrr_npr import DetectLLM
from baseline_models.rank import Rank
from tqdm import tqdm

class GPT2Worker:
    def __init__(self, model_path):
        # Initialize once per actor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True
        )

        # Init detection components
        self.entropy = Entropy(self.model, self.tokenizer)
        self.logp = LogP(self.model, self.tokenizer)
        self.logrank = LogRank(self.model, self.tokenizer)
        self.detectllm = DetectLLM(self.model, self.tokenizer)
        self.rank = Rank(self.model, self.tokenizer)
        self.diversity = Diversity(self.model, self.tokenizer)
    
    def infer(self, text):
        return {
            "entropy": self.entropy.compute_entropy(text),
            "logp": self.logp.compute_log_p(text),
            "logrank": self.logrank.compute_logrank(text),
            "detectllm": [self.detectllm.compute_LRR(text), self.detectllm.compute_NPR(text)],
            "rank": self.rank.compute_rank(text),
            "diversity": self.diversity.compute_features(text)
        }
        
    def infer_multiple(self, texts):
        results = [None] * len(texts)
    
        for i, text in enumerate(tqdm(texts)):
            try:
                entropy = self.entropy.compute_entropy(text)
            except:
                entropy = 0
    
            try:
                logp = self.logp.compute_log_p(text)
            except:
                logp = 0
    
            try:
                logrank = self.logrank.compute_logrank(text)
            except:
                logrank = 0
    
            try:
                detectllm_LRR = self.detectllm.compute_LRR(text)
            except:
                detectllm_LRR = 0
    
            try:
                detectllm_NPR = self.detectllm.compute_NPR(text)
            except:
                detectllm_NPR = 0
    
            try:
                rank = self.rank.compute_rank(text)
            except:
                rank = 0
    
            try:
                diversity = self.diversity.compute_features(text)
            except:
                diversity = 0
    
            results[i] = {
                "entropy": entropy,
                "logp": logp,
                "logrank": logrank,
                "detectllm": [detectllm_LRR, detectllm_NPR],
                "rank": rank,
                "diversity": diversity
            }
    
        return results

    def shutdown(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()