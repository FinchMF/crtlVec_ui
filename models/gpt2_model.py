from .base_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
import torch

class GPT2ControlModel(LanguageModel):
    def __init__(self):
        super().__init__()
        self.base_model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
        self.lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model.eval()
        self.lm_model.eval()

    @torch.no_grad()
    def get_hidden_state(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.base_model(**inputs)
        return outputs.hidden_states[-1][0, -1]

    @torch.no_grad()
    def generate_with_control(self, prompt, control_vector, scale=1.0):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        def hook_fn(module, input, output):
            hidden_states = output[0]
            modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
            return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)

        hook = self.lm_model.transformer.h[-1].register_forward_hook(hook_fn)
        output_ids = self.lm_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=80,
            do_sample=True,
            top_k=40
        )
        hook.remove()

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
