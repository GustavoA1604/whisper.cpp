#!/usr/bin/env python3

import torch
import numpy as np
import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

def print_task(task_name):
    print(f"\n[TASK: {task_name}]")

# Initialize model
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

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

# Test input
sentence = "Hello world"
src_lang = "eng_Latn"
tgt_lang = "hin_Deva"

print_section("INDICTRANS2 VALIDATION FOR C++ IMPLEMENTATION")
print(f"Input: '{sentence}' ({src_lang} -> {tgt_lang})")

# Apply GELU precision patching to match GGML exactly
patch_gelu_activation(model)

# Task 1: Configuration Parameters
print_task("indictrans_config_update")
config = model.config
print(f"encoder_normalize_before: {config.encoder_normalize_before}")
print(f"decoder_normalize_before: {config.decoder_normalize_before}")
print(f"layernorm_embedding: {config.layernorm_embedding}")
print(f"scale_embedding: {config.scale_embedding}")
print(f"encoder_embed_dim: {config.encoder_embed_dim}")
print(f"decoder_embed_dim: {config.decoder_embed_dim}")
print(f"encoder_attention_heads: {config.encoder_attention_heads}")
print(f"decoder_attention_heads: {config.decoder_attention_heads}")
print(f"encoder_ffn_dim: {config.encoder_ffn_dim}")
print(f"decoder_ffn_dim: {config.decoder_ffn_dim}")
print(f"encoder_layers: {config.encoder_layers}")
print(f"decoder_layers: {config.decoder_layers}")

# Task 2: Tokenizer Integration  
print_task("indictrans_tokenizer_integration")
print(f"src_vocab_size: {tokenizer.src_vocab_size}")
print(f"tgt_vocab_size: {tokenizer.tgt_vocab_size}")
print(f"pad_token_id: {tokenizer.pad_token_id}")

# Task 3: Text Preprocessing
print_task("indictrans_preprocessing")
ip = IndicProcessor(inference=True)
processed = ip.preprocess_batch([sentence], src_lang=src_lang, tgt_lang=tgt_lang)
print(f"processed: '{processed[0]}'")

# Tokenization
inputs = tokenizer(processed, return_tensors="pt", padding=True).to(device)
input_ids = inputs.input_ids[0]
print(f"token_ids: {input_ids.tolist()}")
print(f"num_tokens: {len(input_ids)}")

# Task 4: Embeddings
print_task("indictrans_embeddings")
with torch.no_grad():
    encoder = model.get_encoder()
    
    # Raw embeddings
    raw_embeds = encoder.embed_tokens(inputs.input_ids)
    print(f"embed_scale: {encoder.embed_scale}")
    print(f"raw_embeds_shape: {list(raw_embeds.shape)}")
    print(f"raw_embeds_first_5: [{', '.join([f'{x:.6f}' for x in raw_embeds[0, 0, :5].cpu()])}]")
    print(f"raw_embeds_last_5: [{', '.join([f'{x:.6f}' for x in raw_embeds[0, 0, -5:].cpu()])}]")
    
    # Scaled embeddings
    scaled_embeds = raw_embeds * encoder.embed_scale
    print(f"scaled_embeds_first_5: [{', '.join([f'{x:.6f}' for x in scaled_embeds[0, 0, :5].cpu()])}]")
    print(f"scaled_embeds_last_5: [{', '.join([f'{x:.6f}' for x in scaled_embeds[0, 0, -5:].cpu()])}]")
    
    # Layer normalization status
    has_layernorm = encoder.layernorm_embedding is not None
    print(f"has_embedding_layernorm: {has_layernorm}")
    
    # Apply layer normalization to token embeddings (Step 3) - before positional embeddings
    if has_layernorm:
        token_after_layernorm = encoder.layernorm_embedding(scaled_embeds)
        print(f"after_layernorm_first_5: [{', '.join([f'{x:.6f}' for x in token_after_layernorm[0, 0, :5].cpu()])}]")
        print(f"after_layernorm_last_5: [{', '.join([f'{x:.6f}' for x in token_after_layernorm[0, 0, -5:].cpu()])}]")
        # Use the layer-normalized embeddings for further processing
        final_token_embeds = token_after_layernorm
    else:
        final_token_embeds = scaled_embeds

# Task 5: Positional Embeddings
print_task("indictrans_positional_embeddings")
with torch.no_grad():
    pos_embed_layer = encoder.embed_positions
    embed_pos = pos_embed_layer(inputs.input_ids, final_token_embeds)
    
    print(f"pos_embed_type: sinusoidal")
    print(f"padding_idx: {pos_embed_layer.padding_idx}")
    print(f"pos_embed_shape: {list(embed_pos.shape)}")
    
    # Show position encoding for first few tokens (first 5 and last 5 dimensions)
    for i in range(min(3, embed_pos.shape[1])):
        pos_vals_first = embed_pos[0, i, :5].cpu()
        pos_vals_last = embed_pos[0, i, -5:].cpu()
        print(f"token_{i}_pos_first_5: [{', '.join([f'{x:.6f}' for x in pos_vals_first])}]")
        print(f"token_{i}_pos_last_5: [{', '.join([f'{x:.6f}' for x in pos_vals_last])}]")

# Task 6: Combined Embeddings (Step 4 equivalent)
print_task("indictrans_embeddings_combined")
with torch.no_grad():
    # Show the token embeddings before adding positional (should match Step 3)
    print(f"step4_before_pos_first_5: [{', '.join([f'{x:.6f}' for x in final_token_embeds[0, 0, :5].cpu()])}]")
    print(f"step4_before_pos_last_5: [{', '.join([f'{x:.6f}' for x in final_token_embeds[0, 0, -5:].cpu()])}]")
    
    # Show what we're adding (position 0 embeddings)
    pos_0_embeddings = embed_pos[0, 0, :]  # Position 0 embeddings
    print(f"step4_pos0_embeds_first_5: [{', '.join([f'{x:.6f}' for x in pos_0_embeddings[:5].cpu()])}]")
    print(f"step4_pos0_embeds_last_5: [{', '.join([f'{x:.6f}' for x in pos_0_embeddings[-5:].cpu()])}]")
    
    # Show the result after adding positional embeddings (Step 4 output)
    combined = final_token_embeds + embed_pos
    print(f"step4_after_pos_first_5: [{', '.join([f'{x:.6f}' for x in combined[0, 0, :5].cpu()])}]")
    print(f"step4_after_pos_last_5: [{', '.join([f'{x:.6f}' for x in combined[0, 0, -5:].cpu()])}]")

# Task 7: Encoder Layers (just validate structure)
print_task("indictrans_encoder_layers")
print(f"num_encoder_layers: {len(encoder.layers)}")
first_layer = encoder.layers[0]
print(f"self_attn_heads: {first_layer.self_attn.num_heads}")
print(f"self_attn_head_dim: {first_layer.self_attn.head_dim}")
print(f"ffn_activation: {config.activation_function}")
print(f"layer_normalize_before: {first_layer.normalize_before}")

# Task 8: Complete Encoder Forward Pass with Layer-by-Layer Debug
print_task("indictrans_encoder_forward")
with torch.no_grad():
    # Manual layer-by-layer execution for debugging
    print("🔍 ENCODER LAYER DEBUG OUTPUT:")
    
    # Start with combined embeddings
    hidden_states = combined
    
    # Prepare attention mask for manual layer calls
    # Convert 2D mask (1, seq_len) to 4D mask (1, 1, seq_len, seq_len)
    batch_size, seq_len = inputs.attention_mask.shape
    # Create causal mask for encoder (all positions can attend to all positions)
    extended_attention_mask = inputs.attention_mask[:, None, None, :]  # (1, 1, 1, seq_len)
    extended_attention_mask = extended_attention_mask.expand(batch_size, 1, seq_len, seq_len)  # (1, 1, seq_len, seq_len)
    # Convert to attention weights format (0 for attend, large negative for mask)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    # Process first 3 layers with debug output
    for layer_idx in range(min(3, len(encoder.layers))):
        layer = encoder.layers[layer_idx]
        
        # Print input to this layer
        input_first_5 = hidden_states[0, 0, :5].cpu()
        input_last_5 = hidden_states[0, 0, -5:].cpu()
        if layer_idx == 0:
            print(f"Layer {layer_idx} Input  - First 5: [{', '.join([f'{x:.6f}' for x in input_first_5])}]")
            print(f"Layer {layer_idx} Input  - Last 5:  [{', '.join([f'{x:.6f}' for x in input_last_5])}]")
        
        # Pass through this layer with properly formatted attention mask
        hidden_states = layer(
            hidden_states, 
            attention_mask=extended_attention_mask,
            layer_head_mask=None  # No head masking
        )[0]
        
        # Print output from this layer
        output_first_5 = hidden_states[0, 0, :5].cpu()
        output_last_5 = hidden_states[0, 0, -5:].cpu()
        print(f"Layer {layer_idx} Output - First 5: [{', '.join([f'{x:.6f}' for x in output_first_5])}]")
        print(f"Layer {layer_idx} Output - Last 5:  [{', '.join([f'{x:.6f}' for x in output_last_5])}]")
    
    # Process remaining layers without debug output
    for layer_idx in range(3, len(encoder.layers)):
        layer = encoder.layers[layer_idx]
        hidden_states = layer(
            hidden_states, 
            attention_mask=extended_attention_mask,
            layer_head_mask=None  # No head masking
        )[0]
    
    # Apply final layer norm if present
    if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
        hidden_states = encoder.layer_norm(hidden_states)
    
    # Final encoder output
    encoder_hidden = hidden_states[0]  # [seq_len, hidden_dim]
    encoder_flat = encoder_hidden.cpu().numpy().flatten()
    
    final_first_5 = encoder_hidden[0, :5].cpu()
    final_last_5 = encoder_hidden[0, -5:].cpu()
    print(f"Final Encoder - First 5: [{', '.join([f'{x:.6f}' for x in final_first_5])}]")
    print(f"Final Encoder - Last 5:  [{', '.join([f'{x:.6f}' for x in final_last_5])}]")
    
    print(f"encoder_output_shape: {list(encoder_hidden.shape)}")
    print(f"total_elements: {len(encoder_flat)}")
    print(f"First 5 values: [{', '.join([f'{x:.6f}' for x in final_first_5])}]")
    print(f"Last 5 values: [{', '.join([f'{x:.6f}' for x in final_last_5])}]")
    print(f"statistics: min={np.min(encoder_flat):.6f}, max={np.max(encoder_flat):.6f}, mean={np.mean(encoder_flat):.6f}")

# Quick translation validation
print_task("translation_validation")
with torch.no_grad():
    generated = model.generate(**inputs, max_length=50, num_beams=5)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translation = ip.postprocess_batch(decoded, lang=tgt_lang)
    print(f"translation: '{translation[0]}'")

print_section("VALIDATION COMPLETE") 