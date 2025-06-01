from .base_model import LanguageModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

class BertControlModel(LanguageModel):
    def __init__(self):
        super().__init__()
        self.base_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.base_model.eval()
        self.lm_model.eval()

    @torch.no_grad()
    def get_hidden_state(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.base_model(**inputs)
        return outputs.hidden_states[-1][0, 0]  # [CLS] token

    @torch.no_grad()
    def generate_with_control(self, prompt, control_vector, scale=1.0):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        def hook_fn(module, input, output):
            hidden_states = output[0]
            modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
            return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)

        hook = self.lm_model.bert.encoder.layer[-1].register_forward_hook(hook_fn)
        outputs = self.lm_model(**inputs)
        hook.remove()

        mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        if len(mask_token_index) == 0:
            return prompt + " [No mask token found]"
        
        mask_token_logits = outputs.logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0]
        return prompt.replace("[MASK]", self.tokenizer.decode(top_tokens[0]))
