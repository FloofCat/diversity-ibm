import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class Entropy:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.features = []
        
    def compute_entropy(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
        token_log_probs = [log_probs[0, i, token].item() for i, token in enumerate(tokens[0][1:])]
            
        probs = [torch.exp(torch.tensor(log_prob)) for log_prob in token_log_probs]
        if len(probs) == 0: return 0
        entropy = -sum([prob * torch.log(prob) for prob in probs]) / len(probs)
     
        return entropy.item()   
        