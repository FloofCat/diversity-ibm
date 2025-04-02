import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from .diversity import Diversity
from .entropy import Entropy
from .fastdetect import FastDetect
from .logp import LogP
from .logrank import LogRank
from .lrr_npr import DetectLLM
from .rank import Rank
from .roberta import RobertaBase
from .radar import RADAR
from .binoculars import Binoculars
from .raidar import RAIDAR
from .t5sentinel import T5Predictor
from .biscope import BiScope

class Baselines:
    def __init__(self):
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
        
        self.radar_tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        self.radar_model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        
        self.binoculars_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        self.binoculars_observer_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
        self.binoculars_performer_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
        
        self.raidar_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    
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
        self.t5sentinel = T5Predictor("./data/checkpoint/T5Sentinel.0613.pt")
        self.biscope = BiScope(self.binoculars_performer_model, self.binoculars_tokenizer, self.binoculars_performer_model, self.binoculars_tokenizer)
        
        
        # Load Diversity
        self.diversity = Diversity(self.gpt2_tokenizer, self.gpt2_model)
    
# Example testing
baselines = Baselines()