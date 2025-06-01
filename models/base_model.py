from abc import ABC, abstractmethod
import torch

class LanguageModel(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.custom_vectors = {}

    @abstractmethod
    def get_hidden_state(self, prompt):
        pass

    @abstractmethod
    def generate_with_control(self, prompt, control_vector, scale=1.0):
        pass

    def make_control_vector(self, pos_examples, neg_examples):
        vec_pos = torch.stack([self.get_hidden_state(p) for p in pos_examples]).mean(dim=0)
        vec_neg = torch.stack([self.get_hidden_state(p) for p in neg_examples]).mean(dim=0)
        return vec_pos - vec_neg
