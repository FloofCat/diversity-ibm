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
        self.selected_dataset = ('Human', 'ChatGPT', 'PaLM', 'LLaMA', 'GPT2')
        self.label_token_ids = self.get_label_token_ids(self.selected_dataset)
        
    def get_label_token_ids(self, dataset_labels):
        """Maps dataset labels to token IDs using the tokenizer."""
        label_token_ids = {}
        for label in dataset_labels:
            token_id = self.tokenizer.convert_tokens_to_ids(label)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                print(f"Warning: Label '{label}' is unknown in the tokenizer!")
            else:
                label_token_ids[label] = token_id
        return label_token_ids

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
        # Extract scores for the selected dataset labels
        selected_token_ids = list(self.label_token_ids.values())
        filtered_scores = first_token_scores[:, selected_token_ids]  
        
        # Convert to probabilities
        probabilities = F.softmax(filtered_scores, dim=-1)

        # Map probabilities back to their respective labels
        prob_dict = {label: probabilities[0, i].item() for i, label in enumerate(self.selected_dataset)}

        return prob_dict
    
    def del_models(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        