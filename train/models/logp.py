import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class LogP:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.features = []
        
    def compute_log_p(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = [log_probs[0, i, token].item() for i, token in enumerate(tokens[0][1:])]
        
        return sum(token_log_probs) / len(token_log_probs)
        