from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RobertaBase:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
        self.model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
        self.features = []
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=-1)
        return probabilities[0][1].item()