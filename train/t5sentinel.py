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
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs["decoder_input_ids"] = torch.tensor([[self.tokenizer.pad_token_id]], device=self.model.device)

        with torch.no_grad():
            # Get model outputs (logits and hidden states)
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states
        
        # Return probability of AI-generated text
        probabilities = F.log_softmax(logits, dim=-1)
        # Return 1 number
        return probabilities[0][1].item()
    
    def del_models(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        