import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class BiScope:
    def __init__(self, detect_model, detect_tokenizer, summary_model, summary_tokenizer):
        self.detect_model = detect_model
        self.detect_tokenizer = detect_tokenizer
        self.summary_model = summary_model
        self.summary_tokenizer = summary_tokenizer
        self.detect_model.eval()
        self.summary_model.eval()

    def generate_summary(self, text):
        """
        Generate a summary using the summary model if available, or return an empty string.
        """
        if self.summary_model:
            input_text = f"Write a title for this text: {text}\nJust output the title:"
            summary_ids = self.summary_tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True).input_ids.to(self.summary_model.device)
            summary_output = self.summary_model.generate(summary_ids, max_length=1024, num_return_sequences=1, temperature=0.0)
            summary_text = self.summary_tokenizer.decode(summary_output[0], skip_special_tokens=True).strip()
            return summary_text
        return ''

    def compute_loss_features(self, logits, targets, text_slice, loss_type='fce'):
        """
        Compute loss features (FCE or BCE) for the given logits and targets.
        """
        if loss_type == 'fce':
            loss = torch.nn.CrossEntropyLoss(reduction='none')(
                logits[0, text_slice.start-1:text_slice.stop-1, :],
                targets
            )
        else:
            loss = torch.nn.CrossEntropyLoss(reduction='none')(
                logits[0, text_slice, :],
                targets
            )
        
        return loss.detach().cpu().numpy()

    def extract_features(self, text, sample_clip=1024):
        """
        Extract features (FCE and BCE loss) from a single sample.
        """
        # Generate the summary prompt.
        summary_text = self.generate_summary(text)
        prompt_text = f"Complete the following text: {summary_text}"

        # Tokenize the input text and the prompt.
        prompt_ids = self.detect_tokenizer(prompt_text, return_tensors='pt', max_length=sample_clip, truncation=True).input_ids.to(self.detect_model.device)
        text_ids = self.detect_tokenizer(text, return_tensors='pt', max_length=sample_clip, truncation=True).input_ids.to(self.detect_model.device)

        combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
        text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])
        
        outputs = self.detect_model(combined_ids, labels=combined_ids)
        logits = outputs.logits
        targets = combined_ids[0][text_slice]

        # Compute the loss features (FCE and BCE).
        fce_loss = self.compute_loss_features(logits, targets, text_slice, loss_type='fce')
        bce_loss = self.compute_loss_features(logits, targets, text_slice, loss_type='bce')
        
        features = []
        for p in range(1, 10):
            split = len(fce_loss) * p // 10
            features.extend([
                np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
                np.min(fce_loss[split:]), np.std(fce_loss[split:]),
                np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
                np.min(bce_loss[split:]), np.std(bce_loss[split:])
            ])
        return features

    def detect_sample(self, sample):
        return self.extract_features(sample)