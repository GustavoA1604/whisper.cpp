#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

class GGMLGELUActivation(torch.nn.Module):
    """GGML GELU implementation as a proper PyTorch module for precision matching"""
    def __init__(self):
        super().__init__()
        self.GELU_COEF_A = 0.044715
        self.SQRT_2_OVER_PI = 0.79788456080286535587989211986876
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(self.SQRT_2_OVER_PI * x * (1.0 + self.GELU_COEF_A * x * x)))

def patch_gelu_activation(module, visited=None):
    """Patch all GELU activations to match GGML precision"""
    if visited is None:
        visited = set()
    
    module_id = id(module)
    if module_id in visited:
        return
    visited.add(module_id)
    
    for name, child in module.named_children():
        if hasattr(child, 'activation_fn'):
            if hasattr(child.activation_fn, '__class__'):
                class_name = child.activation_fn.__class__.__name__.lower()
                if 'gelu' in class_name and not isinstance(child.activation_fn, GGMLGELUActivation):
                    child.activation_fn = GGMLGELUActivation()
        patch_gelu_activation(child, visited)

# Initialize model
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
patch_gelu_activation(model)
print(f"Model: {model_name}")

# Test input
sentence = "Hello world"
src_lang = "eng_Latn"
tgt_lang = "hin_Deva"
print(f"Input: '{sentence}' ({src_lang} -> {tgt_lang})")

# Setup
ip = IndicProcessor(inference=True)
processed = ip.preprocess_batch([sentence], src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(processed, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].cpu().tolist()[0])
print(f"Tokenizer output IDs: {inputs['input_ids'].cpu().tolist()[0]}")
print(f"Tokenizer output tokens: {tokens}")

with torch.no_grad():
    encoder = model.get_encoder()

    raw_embeds = encoder.embed_tokens(inputs['input_ids'])
    scaled_embeds = raw_embeds * encoder.embed_scale
    
    if encoder.layernorm_embedding is not None:
        scaled_embeds = encoder.layernorm_embedding(scaled_embeds)
    
    seq_len = inputs['input_ids'].size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0) + encoder.padding_idx + 1
    pos_embeds = encoder.embed_positions(positions)
    encoder_input = scaled_embeds + pos_embeds
    
    first_5 = encoder_input[0, 0, :5].cpu()
    last_5 = encoder_input[0, 0, -5:].cpu()
    print(f"Encoder input (first 5 values): [{', '.join([f'{x:.6f}' for x in first_5])}]")
    print(f"Encoder input (last 5 values): [{', '.join([f'{x:.6f}' for x in last_5])}]")
    
    hidden_states = encoder_input
    
    batch_size, seq_len = inputs['attention_mask'].shape
    extended_attention_mask = inputs['attention_mask'][:, None, None, :].expand(batch_size, 1, seq_len, seq_len)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    for layer in encoder.layers:
        hidden_states = layer(hidden_states, attention_mask=extended_attention_mask, layer_head_mask=None)[0]
    
    if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
        hidden_states = encoder.layer_norm(hidden_states)
    
    first_5 = hidden_states[0, 0, :5].cpu()
    last_5 = hidden_states[0, 0, -5:].cpu()
    print(f"Encoder output (first 5 values): [{', '.join([f'{x:.6f}' for x in first_5])}]")
    print(f"Encoder output (last 5 values): [{', '.join([f'{x:.6f}' for x in last_5])}]")

with torch.no_grad():
    generated = model.generate(**inputs, max_length=50, num_beams=5)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translation = ip.postprocess_batch(decoded, lang=tgt_lang)
    print(f"Final translation: '{translation[0]}'") 