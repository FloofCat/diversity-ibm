import torch
import gc
import numpy as np
from train.baseline_models.diversity import Diversity
from train.baseline_models.entropy import Entropy
from train.baseline_models.logp import LogP
from train.baseline_models.logrank import LogRank
from train.baseline_models.lrr_npr import DetectLLM
from train.baseline_models.rank import Rank

class GPT2Worker:
    def __init__(self, model_path):
        # Initialize once per actor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
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

    def shutdown(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()