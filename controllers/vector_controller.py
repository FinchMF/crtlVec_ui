import torch
import json
import os

class VectorController:
    def __init__(self, model_name):
        self.model_name = model_name
        self.vectors = {}
        self.vector_path = f"vectors_{model_name}.json"
        self.load_vectors()

    def add_vector(self, name, vector):
        """Add or update a control vector"""
        self.vectors[name] = vector
        self.save_vectors()

    def get_vector(self, name):
        """Retrieve a control vector"""
        return self.vectors.get(name)

    def save_vectors(self):
        """Save vectors to disk"""
        vector_dict = {name: tensor.tolist() for name, tensor in self.vectors.items()}
        with open(self.vector_path, 'w') as f:
            json.dump(vector_dict, f)

    def load_vectors(self):
        """Load vectors from disk"""
        if os.path.exists(self.vector_path):
            with open(self.vector_path, 'r') as f:
                vector_dict = json.load(f)
                self.vectors = {name: torch.tensor(data) for name, data in vector_dict.items()}
