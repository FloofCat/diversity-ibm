import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

        # Load all models
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/gpt2", use_fast=False, trust_remote_code=True)
        self.gpt2_model = self.types["gpt2"].from_pretrained(f"{self.cache_dir}/gpt2", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
        
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/roberta", use_fast=False, trust_remote_code=True)
        self.roberta_model = self.types["roberta"].from_pretrained(f"{self.cache_dir}/roberta", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
        
        self.radar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/radar", use_fast=False, trust_remote_code=True)
        self.radar_model = self.types["radar"].from_pretrained(f"{self.cache_dir}/radar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
        
        self.binoculars_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/binoculars_observer", use_fast=False, trust_remote_code=True)
        self.binoculars_observer_model = self.types["binoculars_observer"].from_pretrained(f"{self.cache_dir}/binoculars_observer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
        self.binoculars_performer_model = self.types["binoculars_performer"].from_pretrained(f"{self.cache_dir}/binoculars_performer", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
        
        self.raidar_tokenizer = AutoTokenizer.from_pretrained(f"{self.cache_dir}/raidar", use_fast=False, trust_remote_code=True)
        self.raidar_model = self.types["raidar"].from_pretrained(f"{self.cache_dir}/raidar", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True).eval()
    
        # Load all baseline models
        self.entropy = Entropy(self.gpt2_tokenizer, self.gpt2_model)
        self.fastdetect = FastDetect()
        self.logp = LogP(self.gpt2_tokenizer, self.gpt2_model)
        self.logrank = LogRank(self.gpt2_tokenizer, self.gpt2_model)
        self.detectllm = DetectLLM(self.gpt2_tokenizer, self.gpt2_model)
        self.rank = Rank(self.gpt2_tokenizer, self.gpt2_model)
        self.roberta = RobertaBase(self.roberta_tokenizer, self.roberta_model)
        self.radar = RADAR(self.radar_tokenizer, self.radar_model)
        self.binoculars = Binoculars(self.binoculars_observer_model, self.binoculars_performer_model, self.binoculars_tokenizer)
        self.raidar = RAIDAR(self.raidar_tokenizer, self.binoculars_performer_model)
        self.t5sentinel = T5Predictor("./model-cache/T5Sentinel.0613.pt")
        self.biscope = BiScope(self.binoculars_performer_model, self.binoculars_tokenizer, self.binoculars_performer_model, self.binoculars_tokenizer)
        
        
        # Load Diversity
        self.diversity = Diversity(self.gpt2_tokenizer, self.gpt2_model)
    
# Example testing
baselines = Baselines()
sample_text = """
The academic paper titled "FUTURE-AI: Guiding Principles and Consensus Recommendations for Trustworthy Artificial Intelligence in Future Medical Imaging" presents a set of guiding principles and consensus recommendations for the development and implementation of trustworthy artificial intelligence (AI) in medical imaging. The paper emphasizes the importance of AI in improving the accuracy and efficiency of medical diagnosis and treatment, while also acknowledging the potential risks and challenges associated with the use of AI in healthcare. The paper proposes a set of guiding principles and recommendations that aim to ensure the responsible and ethical development and use of AI in medical imaging, including transparency, accountability, and patient-centeredness. Overall, the paper provides valuable insights and guidance for researchers, practitioners, and policymakers involved in the development and implementation of AI in medical imaging.
"""

# Example usage of the models
print("Entropy Score:", baselines.entropy.compute_entropy(sample_text))
print("FastDetect Score:", baselines.fastdetect.detect(sample_text))
print("LogP Score:", baselines.logp.compute_log_p(sample_text))
print("LogRank Score:", baselines.logrank.compute_logrank(sample_text))
print("DetectLLM Score:", baselines.detectllm.compute_log_probs_ranks(sample_text))
print("Rank Score:", baselines.rank.compute_rank(sample_text))
print("Roberta Score:", baselines.roberta.predict(sample_text))
print("RADAR Score:", baselines.radar.detect_probability(sample_text))
print("Binoculars Score:", baselines.binoculars.compute_score(sample_text))
print("RAIDAR Score:", baselines.raidar.compute_raidar(sample_text))
print("T5Sentinel Score:", baselines.t5sentinel.compute_t5(sample_text))
print("BiScope Score:", baselines.biscope.extract_features(sample_text))
print("Diversity Score:", baselines.diversity.compute_features(sample_text))
