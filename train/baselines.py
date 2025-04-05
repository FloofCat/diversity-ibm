import torch
import gc
import json
import pandas as pd
import numpy as np
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from diversity import Diversity
from entropy import Entropy
from fastdetect import FastDetect
from logp import LogP
from logrank import LogRank
from lrr_npr import DetectLLM
from rank import Rank
from roberta import RobertaBase
from radar import RADAR
from binoculars import Binoculars
from raidar import RAIDAR
from t5sentinel import T5Predictor
from biscope import BiScope
from downloader import Downloader

class Baselines:
    def __init__(self):
        # Download models if not already downloaded
        self.cache_dir = "./model-cache"
        self.models = {
            "gpt2": "gpt2",
            "roberta": "openai-community/roberta-base-openai-detector",
            "radar": "TrustSafeAI/RADAR-Vicuna-7B",
            "binoculars_observer": "tiiuae/falcon-7b",
            "binoculars_performer": "tiiuae/falcon-7b-instruct",
            "raidar": "tiiuae/falcon-7b-instruct"
        }
        self.types = {
            "gpt2": AutoModelForCausalLM,
            "roberta": AutoModelForSequenceClassification,
            "radar": AutoModelForSequenceClassification,
            "binoculars_observer": AutoModelForCausalLM,
            "binoculars_performer": AutoModelForCausalLM,
            "raidar": AutoModelForCausalLM
        }

        self.downloader = Downloader(self.models, self.types, self.cache_dir)
        self.no_threads = 4
    
    def detect_gpt2(self, texts):
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/gpt2", use_fast=False, trust_remote_code=True)
        self.gpt2_model = self.types["gpt2"].from_pretrained(f"{self.cache_dir}/gpt2", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] Loaded GPT2 model.")

        # self.entropy = Entropy(self.gpt2_model, self.gpt2_tokenizer)
        # self.logp = LogP(self.gpt2_model, self.gpt2_tokenizer)
        # self.logrank = LogRank(self.gpt2_model, self.gpt2_tokenizer)
        # self.detectllm = DetectLLM(self.gpt2_model, self.gpt2_tokenizer)
        # self.rank = Rank(self.gpt2_model, self.gpt2_tokenizer)
        # self.diversity = Diversity(self.gpt2_model, self.gpt2_tokenizer)

        # chunk_size = len(texts) // self.no_threads
        # results = [None] * len(texts)
        # threads = []

        # def _detect(start_idx, end_idx, text_chunk):
        #     for i, text in enumerate(text_chunk):
        #         features = {
        #             "entropy": self.entropy.compute_entropy(text),
        #             "logp": self.logp.compute_log_p(text),
        #             "logrank": self.logrank.compute_logrank(text),
        #             "detectllm": [self.detectllm.compute_LRR(text), self.detectllm.compute_NPR(text)],
        #             "rank": self.rank.compute_rank(text),
        #             "diversity": self.diversity.compute_features(text)
        #         }
            
        #         results[start_idx + i] = features
        #     print(f"[THREAD] Thread {start_idx} to {end_idx} has completed their tasks.")            
        
        # for i in range(self.no_threads):
        #     start_idx = i * chunk_size
        #     end_idx = len(texts) if i == self.no_threads - 1 else (i + 1) * chunk_size
        #     text_chunk = texts[start_idx:end_idx]
        #     thread = threading.Thread(target=_detect, args=(start_idx, end_idx, text_chunk))
        #     threads.append(thread)
        #     thread.start()

        # for thread in threads:
        #     thread.join()

        del self.gpt2_tokenizer
        del self.gpt2_model
        gc.collect()
        torch.cuda.empty_cache()
        print("[LOGS] GPT2 Completed")

        return results
    
    def detect_roberta(self, text):
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/roberta", use_fast=False, trust_remote_code=True)
        self.roberta_model = self.types["roberta"].from_pretrained(f"{self.cache_dir}/roberta", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] Roberta Loaded")
        self.roberta = RobertaBase(self.roberta_model, self.roberta_tokenizer)

        features = {
            "roberta": self.roberta.predict(text)
        }

        del self.roberta_tokenizer
        del self.roberta_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] Roberta Completed")

        return features

    def detect_radar(self, text):
        self.radar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/radar", use_fast=False, trust_remote_code=True)
        self.radar_model = self.types["radar"].from_pretrained(f"{self.cache_dir}/radar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] RADAR Loaded")

        self.radar = RADAR(self.radar_model, self.radar_tokenizer)

        features = {
            "radar": self.radar.detect_probability(text)
        }

        del self.radar_tokenizer
        del self.radar_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] RADAR Completed")

        return features

    def detect_binoculars_biscope(self, text):
        self.binoculars_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/binoculars_observer", use_fast=False, trust_remote_code=True)
        self.binoculars_observer_model = self.types["binoculars_observer"].from_pretrained(f"{self.cache_dir}/binoculars_observer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        self.binoculars_performer_model = self.types["binoculars_performer"].from_pretrained(f"{self.cache_dir}/binoculars_performer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] Binoculars Loaded")

        self.binoculars = Binoculars(self.binoculars_observer_model, self.binoculars_performer_model, self.binoculars_tokenizer)
        self.biscope = BiScope(self.binoculars_performer_model, self.binoculars_tokenizer, self.binoculars_performer_model, self.binoculars_tokenizer)

        features = {
            "binoculars": self.binoculars.compute_score(text),
            "biscope": self.biscope.extract_features(text)
        }

        del self.binoculars_tokenizer
        del self.binoculars_performer_model
        del self.binoculars_observer_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] Binoculars Completed")

        return features
    
    def detect_raidar(self, text):
        self.raidar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/raidar", use_fast=False, trust_remote_code=True)
        self.raidar_model = self.types["raidar"].from_pretrained(f"{self.cache_dir}/raidar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        print("[LOGS] RAIDAR Loaded")

        self.raidar = RAIDAR(self.raidar_model, self.raidar_tokenizer)

        features = {
            "raidar": self.raidar.compute_raidar(text)
        }

        del self.raidar_tokenizer
        del self.raidar_model
        gc.collect()
        torch.cuda.empty_cache() 
        print("[LOGS] RAIDAR Completed")

        return features

    def detect_others(self, text):
        self.fastdetect = FastDetect()
        self.t5sentinel = T5Predictor("./model-cache/T5Sentinel.0613.pt")
        print("[LOGS] Others Loaded")

        features = {
            "fastdetect": self.fastdetect.detect(text),
            "t5sentinel": self.t5sentinel.compute_t5(text)
        }

        self.t5sentinel.del_models()
        print("[LOGS] Others completed")

        return features

    def detect(self, text):
        features = {}

        # Run all detection methods and merge results
        features.update(self.detect_gpt2(text))
        features.update(self.detect_roberta(text))
        features.update(self.detect_radar(text))
        features.update(self.detect_binoculars_biscope(text))
        features.update(self.detect_raidar(text))
        features.update(self.detect_others(text))

        return features

    
# Example testing
print("What is happening? Has this docker image actually imported yet?")
baselines = Baselines()
# sample_text = """
# The academic paper titled "FUTURE-AI: Guiding Principles and Consensus Recommendations for Trustworthy Artificial Intelligence in Future Medical Imaging" presents a set of guiding principles and consensus recommendations for the development and implementation of trustworthy artificial intelligence (AI) in medical imaging. The paper emphasizes the importance of AI in improving the accuracy and efficiency of medical diagnosis and treatment, while also acknowledging the potential risks and challenges associated with the use of AI in healthcare. The paper proposes a set of guiding principles and recommendations that aim to ensure the responsible and ethical development and use of AI in medical imaging, including transparency, accountability, and patient-centeredness. Overall, the paper provides valuable insights and guidance for researchers, practitioners, and policymakers involved in the development and implementation of AI in medical imaging.
# """

# print(baselines.detect(sample_text))
train_df = pd.read_csv(baselines.cache_dir + "/raid/train.csv")
texts = train_df["generation"][:5000].tolist()

# features = baselines.detect_gpt2(texts)
# print(len(features))