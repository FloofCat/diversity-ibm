from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import login
import os


class Downloader:
    def __init__(self, models, types, cache_dir):
        self.models = models
        self.types = types
        self.cache_dir = cache_dir

        # login(token="XXXXXXXXXXXXXXXXXX")
        self.download_models()

    def download_models(self):
        for model_name, model_id in self.models.items():
            print(f"Downloading {model_name}...")

            # Check if the model is already downloaded
            if(os.path.exists(f"{self.cache_dir}/{model_name}")):
                print(f"{model_name} already downloaded.")
                continue
            

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            model = self.types[model_name].from_pretrained(model_id,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True, use_cache=False)

            model.save_pretrained(f"{self.cache_dir}/{model_name}")
            tokenizer.save_pretrained(f"{self.cache_dir}/{model_name}")

            # Unload the model
            del model
            del tokenizer
            torch.cuda.empty_cache()

            print(f"{model_name} downloaded and saved to {self.cache_dir}/{model_name}")
