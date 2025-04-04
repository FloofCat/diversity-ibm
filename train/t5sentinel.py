import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch.nn.functional as F
import gc


class T5Predictor:
    def __init__(self, model_path: str, backbone_name: str = "t5-small"):
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(backbone_name)
        self.model.load_state_dict(torch.load(model_path)["model"], strict=False)
        self.model.eval()
        self.tokenizer = T5TokenizerFast.from_pretrained(backbone_name)

    def compute_t5(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=2,  # first token = label, second = EOS
                output_scores=True,
                return_dict_in_generate=True
            )

        # Get the logits of the first generated token (label token)
        first_token_scores = outputs.scores[0]  # [batch_size, vocab_size]
        probs = F.softmax(first_token_scores, dim=-1)  # Get actual probabilities
        # Return 1 number
        return probs
    
    def del_models(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        