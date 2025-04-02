import torch
import Levenshtein

class RAIDAR:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.features = []
    
    def rewrite_text(self, input_text):
        system_prompt = "Concise this for me and keep all the information:"
        prompt = f"{system_prompt}\n\n{input_text}"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        output_ids = self.model.generate(input_ids, max_length=1024, num_return_sequences=1, temperature=0.0)
        rewritten_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        return rewritten_text
    
    def compute_raidar(self, original, rewritten):
        lev_distance = Levenshtein.distance(original, rewritten)
        max_len = max(len(original), len(rewritten))
        return 1 - (lev_distance / max_len) if max_len > 0 else 1        