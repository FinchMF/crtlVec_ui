import gradio as gr
from transformers import (GPT2Tokenizer, GPT2LMHeadModel, GPT2Model,
                        BertTokenizer, BertModel, BertForMaskedLM)
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image  # Add this import

# Load GPT-2 models
base_model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load BERT models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_base_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Add this after model loading at the start
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model.eval()
lm_model.eval()

bert_base_model.eval()
bert_lm_model.eval()

# Predefined prompt groups for default control types
prompt_sets = {
    "Sentiment": {
        "positive": ["The movie was fantastic.", "I loved the experience."],
        "negative": ["The movie was terrible.", "I hated the experience."]
    },
    "Formality": {
        "formal": ["I am writing to inform you.", "Please find the report attached."],
        "informal": ["Hey! Just letting you know.", "Check this out!"]
    },
    "Tense": {
        "present": ["He walks to school.", "She eats lunch."],
        "past": ["He walked to school.", "She ate lunch."]
    }
}

# BERT-specific prompt sets
bert_prompt_sets = {
    "Sentiment": {
        "positive": ["This is wonderful.", "I am very happy."],
        "negative": ["This is terrible.", "I am very sad."]
    },
    "Formality": {
        "formal": ["Dear Sir/Madam,", "As per our discussion,"],
        "informal": ["Hey there!", "What's up?"]
    }
}

# Separate storage for custom vectors
gpt2_custom_vectors = {}
bert_custom_vectors = {}

# Utility to extract last token hidden state
@torch.no_grad()
def get_last_hidden(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = base_model(**inputs)
    return outputs.hidden_states[-1][0, -1]  # [batch=0, last_token]

# Create control vector from two prompt groups
def make_control_vector(group):
    pairs = prompt_sets[group]
    vec_a = torch.stack([get_last_hidden(p) for p in pairs[list(pairs.keys())[0]]]).mean(dim=0)
    vec_b = torch.stack([get_last_hidden(p) for p in pairs[list(pairs.keys())[1]]]).mean(dim=0)
    return vec_a - vec_b

# Generate text with a control vector injected
@torch.no_grad()
def generate_with_control(prompt, control_vector, scale=1.0):
    # Properly tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    def hook_fn(module, input, output):
        hidden_states = output[0]  # Extract hidden states from tuple
        modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
        # Return tuple with modified hidden states and original attention (if present)
        if len(output) == 1:
            return (modified_states,)
        return (modified_states,) + output[1:]

    hook = lm_model.transformer.h[-1].register_forward_hook(hook_fn)
    output_ids = lm_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        max_length=80,
        do_sample=True,
        top_k=40
    )
    hook.remove()
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# BERT-specific utilities
# Utility to extract [CLS] token hidden state
@torch.no_grad()
def get_bert_hidden(prompt):
    inputs = bert_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_base_model(**inputs)
    return outputs.hidden_states[-1][0, 0]  # Get [CLS] token embedding

def make_bert_control_vector(group):
    pairs = bert_prompt_sets[group]
    vec_a = torch.stack([get_bert_hidden(p) for p in pairs[list(pairs.keys())[0]]]).mean(dim=0)
    vec_b = torch.stack([get_bert_hidden(p) for p in pairs[list(pairs.keys())[1]]]).mean(dim=0)
    return vec_a - vec_b

@torch.no_grad()
def generate_with_bert_control(prompt, control_vector, scale=1.0):
    inputs = bert_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    def hook_fn(module, input, output):
        hidden_states = output[0]
        modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
        return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)

    hook = bert_lm_model.bert.encoder.layer[-1].register_forward_hook(hook_fn)
    outputs = bert_lm_model(**inputs)
    hook.remove()

    # Get top prediction for masked token
    mask_token_index = torch.where(inputs.input_ids == bert_tokenizer.mask_token_id)[1]
    if len(mask_token_index) == 0:
        return prompt + " [No mask token found]"
    
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0]
    return prompt.replace("[MASK]", bert_tokenizer.decode(top_tokens[0]))

# Create a PCA plot from prompt group
def create_pca_plot(groups):
    if isinstance(groups, str):
        groups = [groups]
    
    plt.figure(figsize=(7, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    all_components = []
    for group_idx, group in enumerate(groups):
        if group not in prompt_sets:
            continue
            
        pairs = prompt_sets[group]
        labels, vectors = [], []
        for label, prompts in pairs.items():
            for p in prompts:
                hidden = get_last_hidden(p)
                vector_np = hidden.cpu().detach().numpy()
                vectors.append(vector_np)
                labels.append(label)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(vectors)
        all_components.extend(components)
        
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(components[indices, 0], components[indices, 1], 
                       label=f"{group}-{label}",
                       color=colors[group_idx],
                       alpha=0.7,
                       marker=['o', 's'][list(pairs.keys()).index(label)])
    
    if all_components:
        all_components = np.vstack(all_components)
        plt.xlim(all_components[:, 0].min() * 1.2, all_components[:, 0].max() * 1.2)
        plt.ylim(all_components[:, 1].min() * 1.2, all_components[:, 1].max() * 1.2)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"PCA of Prompt Embeddings\n({', '.join(groups)})")
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Gradio interface logic
def interface_fn(prompt, strength, selected_vectors):
    vectors = [make_control_vector(v) if v in prompt_sets else gpt2_custom_vectors[v] for v in selected_vectors]
    combined_vector = sum(vectors)
    output = generate_with_control(prompt, combined_vector, strength)
    # Pass all selected vectors that are in prompt_sets
    plot_vectors = [v for v in selected_vectors if v in prompt_sets]
    plot_buf = create_pca_plot(plot_vectors) if plot_vectors else None
    return output, plot_buf

def bert_interface_fn(prompt, strength, selected_vectors):
    vectors = [make_bert_control_vector(v) if v in bert_prompt_sets else bert_custom_vectors[v] 
              for v in selected_vectors]
    combined_vector = sum(vectors)
    output = generate_with_bert_control(prompt, combined_vector, strength)
    plot_vectors = [v for v in selected_vectors if v in bert_prompt_sets]
    plot_buf = create_pca_plot(plot_vectors) if plot_vectors else None
    return output, plot_buf

# Save custom control vector
def save_custom_vector(name, pos_prompts, neg_prompts):
    try:
        pos_list = [p.strip() for p in pos_prompts.split("\n") if p.strip()]
        neg_list = [p.strip() for p in neg_prompts.split("\n") if p.strip()]
        vec_pos = torch.stack([get_last_hidden(p) for p in pos_list]).mean(dim=0)
        vec_neg = torch.stack([get_last_hidden(p) for p in neg_list]).mean(dim=0)
        gpt2_custom_vectors[name] = vec_pos - vec_neg
        return f"Custom vector '{name}' saved!"
    except Exception as e:
        return f"Error: {e}"

# Save custom vector for BERT
def save_bert_custom_vector(name, pos_prompts, neg_prompts):
    try:
        pos_list = [p.strip() for p in pos_prompts.split("\n") if p.strip()]
        neg_list = [p.strip() for p in neg_prompts.split("\n") if p.strip()]
        vec_pos = torch.stack([get_bert_hidden(p) for p in pos_list]).mean(dim=0)
        vec_neg = torch.stack([get_bert_hidden(p) for p in neg_list]).mean(dim=0)
        bert_custom_vectors[name] = vec_pos - vec_neg
        return f"Custom BERT vector '{name}' saved!"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎛️ Latent Control Vector Generator with Custom Vector Support")

    with gr.Tab("GPT-2"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="The food was")
            strength = gr.Slider(label="Control Strength", minimum=-2, maximum=2, step=0.1, value=1.0)
        vector_type = gr.CheckboxGroup(choices=list(prompt_sets.keys()) + list(gpt2_custom_vectors.keys()),
                                       label="Control Vectors", value=["Sentiment"])
        with gr.Row():
            output = gr.Textbox(label="Generated Output")
            pca_plot = gr.Image(label="PCA Visualization (First Vector)")
        run_button = gr.Button("Generate with Control")
        run_button.click(fn=interface_fn, inputs=[prompt, strength, vector_type], outputs=[output, pca_plot])

    with gr.Tab("BERT"):
        with gr.Row():
            bert_prompt = gr.Textbox(label="Prompt (use [MASK] token)", value="This movie is [MASK].")
            bert_strength = gr.Slider(label="Control Strength", minimum=-2, maximum=2, step=0.1, value=1.0)
        bert_vector_type = gr.CheckboxGroup(choices=list(bert_prompt_sets.keys()) + list(bert_custom_vectors.keys()),
                                          label="Control Vectors", value=["Sentiment"])
        with gr.Row():
            bert_output = gr.Textbox(label="Generated Output")
            bert_pca_plot = gr.Image(label="PCA Visualization")
        bert_run_button = gr.Button("Generate with Control")
        bert_run_button.click(fn=bert_interface_fn, 
                            inputs=[bert_prompt, bert_strength, bert_vector_type],
                            outputs=[bert_output, bert_pca_plot])

    with gr.Tab("Create Custom Vector"):
        vector_name = gr.Textbox(label="Custom Vector Name")
        pos_prompts = gr.Textbox(label="Positive Examples (one per line)")
        neg_prompts = gr.Textbox(label="Negative Examples (one per line)")
        save_output = gr.Textbox(label="Status")
        save_button = gr.Button("Save Custom Vector")
        def wrapped_save(name, pos, neg):
            msg = save_custom_vector(name, pos, neg)
            vector_type.choices = list(prompt_sets.keys()) + list(gpt2_custom_vectors.keys())  # Update list
            return msg
        save_button.click(fn=wrapped_save, inputs=[vector_name, pos_prompts, neg_prompts], outputs=save_output)

        bert_vector_name = gr.Textbox(label="Custom BERT Vector Name")
        bert_pos_prompts = gr.Textbox(label="Positive Examples (one per line)")
        bert_neg_prompts = gr.Textbox(label="Negative Examples (one per line)")
        bert_save_output = gr.Textbox(label="Status")
        bert_save_button = gr.Button("Save Custom BERT Vector")
        def wrapped_save_bert(name, pos, neg):
            msg = save_bert_custom_vector(name, pos, neg)
            bert_vector_type.choices = list(bert_prompt_sets.keys()) + list(bert_custom_vectors.keys())  # Update list
            return msg
        bert_save_button.click(fn=wrapped_save_bert, inputs=[bert_vector_name, bert_pos_prompts, bert_neg_prompts], outputs=bert_save_output)

    gr.Markdown("Use the 'GPT-2' tab to apply GPT-2 control vectors, the 'BERT' tab for BERT control vectors, or the 'Create Custom Vector' tab to define your own from labeled prompts.")

if __name__ == "__main__":
    demo.launch(server_port=7860)