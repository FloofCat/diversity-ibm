import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class LogRank:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.features = []
        
    def compute_logrank(self, text):
        # Compute the log-rank of the text
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1).cuda()

        
        sorted_probs, indices = torch.sort(log_probs, descending=True)
        ranks = [(indices[0, i] == tokens[0][i + 1]).nonzero(as_tuple=True)[0].item() + 1 for i in range(len(tokens[0]) - 1)]
        
        log_ranks = [torch.log(torch.tensor(rank, dtype=torch.float32)).item() for rank in ranks]
        
        return sum(log_ranks) / len(log_ranks)