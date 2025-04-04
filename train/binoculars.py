import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class Binoculars:
    def __init__(self, observer_model, performer_model, tokenizer):
        self.observer_model = observer_model
        self.performer_model = performer_model
        self.tokenizer = tokenizer
        self.observer_model.eval()
        self.performer_model.eval()
    
    def compute_score(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            observer_logits = self.observer_model(tokens).logits
            performer_logits = self.performer_model(tokens).logits
        
        ppl = self._perplexity(tokens, performer_logits)
        x_ppl = self._cross_perplexity(observer_logits, performer_logits, tokens)
        binoculars_score = ppl / x_ppl
        
        return binoculars_score.item()
    
    def _perplexity(self, tokens, logits):
        log_probs = torch.log_softmax(logits.float(), dim=-1).cuda()
        token_log_probs = log_probs[0, torch.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        perplexity = torch.exp(-torch.mean(token_log_probs))
        return perplexity.item()
    
    def _cross_perplexity(self, observer_logits, performer_logits, tokens):
        observer_log_probs = torch.log_softmax(observer_logits.float(), dim=-1).cuda()
        performer_probs = torch.softmax(performer_logits.float(), dim=-1).cuda()
        cross_entropy = -torch.sum(performer_probs[0, torch.arange(len(tokens[0]) - 1), tokens[0, 1:]] * 
                                   observer_log_probs[0, torch.arange(len(tokens[0]) - 1), tokens[0, 1:]])
        cross_entropy /= len(tokens[0]) - 1  # Normalize
        return torch.exp(cross_entropy).item()
