import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class DetectLLM:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.features = []
        
    def compute_log_probs_ranks(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = [log_probs[0, i, token].item() for i, token in enumerate(tokens[0][1:])]
        
        sorted_probs, indices = torch.sort(log_probs, descending=True)
        ranks = [(indices[0, i] == tokens[0][i + 1]).nonzero(as_tuple=True)[0].item() + 1 for i in range(len(tokens[0]) - 1)]
        
        return token_log_probs, ranks
        
    def perturbation_func(self, text, n=5):
        # Make small perturbations to each word in the text
        words = text.split()
        
        for i in range(n):
            idx = random.choice(range(len(words)))
            word = words[idx]
            
            if len(word) > 1:
                char_idx = random.choice(range(len(word)))
                words[idx] = word[:char_idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[char_idx + 1:]
        
        return ' '.join(words)
    
    def compute_LRR(self, text):
        token_log_probs, ranks = self.compute_log_probs_ranks(text)
        log_ranks = [torch.log(torch.tensor(rank, dtype=torch.float32)).item() for rank in ranks]
        LRR = -sum(token_log_probs) / sum(log_ranks)
        return LRR
        
    def compute_NPR(self, text, n=5):
        token_log_probs, ranks = self.compute_log_probs_ranks(text)
        log_rank_original = sum(torch.log(torch.tensor(ranks, dtype=torch.float32))).item()
            
        perturbed_log_ranks = []
        for _ in range(n):
            perturbed_text = self.perturbation_func(text)
            _, perturbed_ranks = self.compute_log_probs_ranks(perturbed_text)
            perturbed_log_ranks.append(sum(torch.log(torch.tensor(perturbed_ranks, dtype=torch.float32))).item())
        
        NPR = sum(perturbed_log_ranks) / (n * log_rank_original)
        return NPR