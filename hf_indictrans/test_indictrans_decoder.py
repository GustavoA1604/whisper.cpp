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

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
patch_gelu_activation(model)

sentence = "Hello world"
src_lang = "eng_Latn"
tgt_lang = "hin_Deva"

print_section("INDICTRANS2 DECODER VALIDATION FOR C++ IMPLEMENTATION")
print(f"Input: '{sentence}' ({src_lang} -> {tgt_lang})")

print_task("encoder_processing")
ip = IndicProcessor(inference=True)
processed = ip.preprocess_batch([sentence], src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(processed, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    encoder = model.get_encoder()

    raw_embeds = encoder.embed_tokens(inputs.input_ids)
    scaled_embeds = raw_embeds * encoder.embed_scale
    
    if encoder.layernorm_embedding is not None:
        token_after_layernorm = encoder.layernorm_embedding(scaled_embeds)
        final_token_embeds = token_after_layernorm
    else:
        final_token_embeds = scaled_embeds
    
    embed_pos = encoder.embed_positions(inputs.input_ids, final_token_embeds)
    combined = final_token_embeds + embed_pos
    
    hidden_states = combined
    batch_size, seq_len = inputs.attention_mask.shape
    extended_attention_mask = inputs.attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.expand(batch_size, 1, seq_len, seq_len)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    for layer in encoder.layers:
        hidden_states = layer(hidden_states, attention_mask=extended_attention_mask, layer_head_mask=None)[0]
    
    if hasattr(encoder, 'layer_norm') and encoder.layer_norm is not None:
        hidden_states = encoder.layer_norm(hidden_states)
    
    encoder_hidden_states = hidden_states
    encoder_attention_mask = inputs.attention_mask

print(f"encoder_output_shape: {list(encoder_hidden_states.shape)}")
print(f"encoder_first_5: [{', '.join([f'{x:.6f}' for x in encoder_hidden_states[0, 0, :5].cpu()])}]")
print(f"encoder_last_5: [{', '.join([f'{x:.6f}' for x in encoder_hidden_states[0, 0, -5:].cpu()])}]")

print_task("decoder_initialization")
decoder = model.get_decoder()
config = model.config

bos_token_id = config.decoder_start_token_id if hasattr(config, 'decoder_start_token_id') else config.bos_token_id
if bos_token_id is None:
    bos_token_id = tokenizer.bos_token_id

print(f"bos_token_id: {bos_token_id}")
print(f"decoder_embed_dim: {config.decoder_embed_dim}")
print(f"decoder_vocab_size: {config.decoder_vocab_size}")
print(f"decoder_layers: {config.decoder_layers}")
print(f"decoder_attention_heads: {config.decoder_attention_heads}")
print(f"decoder_normalize_before: {config.decoder_normalize_before}")
print(f"layernorm_embedding: {config.layernorm_embedding}")
print(f"scale_embedding: {config.scale_embedding}")

decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
print(f"decoder_input_ids: {decoder_input_ids.tolist()}")

print_task("decoder_embeddings")
with torch.no_grad():
    decoder_embed_tokens = decoder.embed_tokens(decoder_input_ids)
    decoder_embed_scale = decoder.embed_scale
    
    print(f"decoder_embed_scale: {decoder_embed_scale}")
    print(f"raw_decoder_embeds_shape: {list(decoder_embed_tokens.shape)}")
    print(f"raw_decoder_embeds_first_5: [{', '.join([f'{x:.6f}' for x in decoder_embed_tokens[0, 0, :5].cpu()])}]")
    print(f"raw_decoder_embeds_last_5: [{', '.join([f'{x:.6f}' for x in decoder_embed_tokens[0, 0, -5:].cpu()])}]")
    
    decoder_scaled_embeds = decoder_embed_tokens * decoder_embed_scale
    print(f"scaled_decoder_embeds_first_5: [{', '.join([f'{x:.6f}' for x in decoder_scaled_embeds[0, 0, :5].cpu()])}]")
    print(f"scaled_decoder_embeds_last_5: [{', '.join([f'{x:.6f}' for x in decoder_scaled_embeds[0, 0, -5:].cpu()])}]")

print_task("decoder_positional_embeddings")
with torch.no_grad():
    decoder_positions = decoder.embed_positions(decoder_input_ids, decoder_scaled_embeds, past_key_values_length=0)
    
    print(f"decoder_pos_embed_shape: {list(decoder_positions.shape)}")
    print(f"decoder_pos_first_5: [{', '.join([f'{x:.6f}' for x in decoder_positions[0, 0, :5].cpu()])}]")
    print(f"decoder_pos_last_5: [{', '.join([f'{x:.6f}' for x in decoder_positions[0, 0, -5:].cpu()])}]")
    
    decoder_combined = decoder_scaled_embeds + decoder_positions
    print(f"decoder_combined_first_5: [{', '.join([f'{x:.6f}' for x in decoder_combined[0, 0, :5].cpu()])}]")
    print(f"decoder_combined_last_5: [{', '.join([f'{x:.6f}' for x in decoder_combined[0, 0, -5:].cpu()])}]")
    
    has_decoder_layernorm = decoder.layernorm_embedding is not None
    print(f"has_decoder_layernorm: {has_decoder_layernorm}")
    
    if has_decoder_layernorm:
        decoder_after_layernorm = decoder.layernorm_embedding(decoder_combined)
        print(f"decoder_after_layernorm_first_5: [{', '.join([f'{x:.6f}' for x in decoder_after_layernorm[0, 0, :5].cpu()])}]")
        print(f"decoder_after_layernorm_last_5: [{', '.join([f'{x:.6f}' for x in decoder_after_layernorm[0, 0, -5:].cpu()])}]")
        decoder_hidden_states = decoder_after_layernorm
    else:
        decoder_hidden_states = decoder_combined

print_task("decoder_attention_masks")
with torch.no_grad():
    decoder_seq_len = decoder_input_ids.size(1)
    
    decoder_attention_mask = torch.ones((1, decoder_seq_len), dtype=torch.long, device=device)
    print(f"decoder_attention_mask: {decoder_attention_mask.tolist()}")
    
    causal_mask = torch.triu(torch.ones((decoder_seq_len, decoder_seq_len), dtype=torch.float, device=device), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, -10000.0)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    print(f"causal_mask_shape: {list(causal_mask.shape)}")
    print(f"causal_mask: {causal_mask[0, 0].tolist()}")
    
    encoder_seq_len = encoder_attention_mask.size(1)
    cross_attention_mask = encoder_attention_mask[:, None, None, :]
    cross_attention_mask = cross_attention_mask.expand(1, 1, decoder_seq_len, encoder_seq_len)
    cross_attention_mask = (1.0 - cross_attention_mask) * -10000.0
    print(f"cross_attention_mask_shape: {list(cross_attention_mask.shape)}")

print_task("decoder_layers_processing")
with torch.no_grad():
    print("🔍 DECODER LAYER DETAILED DEBUG OUTPUT:")
    
    current_hidden_states = decoder_hidden_states
    
    # Process first layer with detailed sub-layer debugging
    layer_idx = 0
    layer = decoder.layers[layer_idx]
    
    input_first_5 = current_hidden_states[0, 0, :5].cpu()
    input_last_5 = current_hidden_states[0, 0, -5:].cpu()
    print(f"Layer {layer_idx} Input  - First 5: [{', '.join([f'{x:.6f}' for x in input_first_5])}]")
    print(f"Layer {layer_idx} Input  - Last 5:  [{', '.join([f'{x:.6f}' for x in input_last_5])}]")
    
    # SUBLAYER 1: Self-Attention Block
    print(f"\n--- Layer {layer_idx} Self-Attention Block ---")
    residual_self = current_hidden_states
    
    # Pre-layer norm for self-attention
    if layer.normalize_before:
        hidden_states_norm = layer.self_attn_layer_norm(current_hidden_states)
    else:
        hidden_states_norm = current_hidden_states
    
    norm_first_5 = hidden_states_norm[0, 0, :5].cpu()
    norm_last_5 = hidden_states_norm[0, 0, -5:].cpu()
    print(f"After self-attn norm - First 5: [{', '.join([f'{x:.6f}' for x in norm_first_5])}]")
    print(f"After self-attn norm - Last 5:  [{', '.join([f'{x:.6f}' for x in norm_last_5])}]")
    
    # Self-attention
    hidden_states_self, _, _ = layer.self_attn(
        hidden_states=hidden_states_norm,
        past_key_value=None,
        attention_mask=causal_mask,
        layer_head_mask=None,
        output_attentions=False,
    )
    
    attn_first_5 = hidden_states_self[0, 0, :5].cpu()
    attn_last_5 = hidden_states_self[0, 0, -5:].cpu()
    print(f"After self-attention - First 5: [{', '.join([f'{x:.6f}' for x in attn_first_5])}]")
    print(f"After self-attention - Last 5:  [{', '.join([f'{x:.6f}' for x in attn_last_5])}]")
    
    # Residual connection
    hidden_states_self = residual_self + hidden_states_self
    
    # Post-layer norm for self-attention (if not normalize_before)
    if not layer.normalize_before:
        hidden_states_self = layer.self_attn_layer_norm(hidden_states_self)
    
    self_first_5 = hidden_states_self[0, 0, :5].cpu()
    self_last_5 = hidden_states_self[0, 0, -5:].cpu()
    print(f"After self-attn block - First 5: [{', '.join([f'{x:.6f}' for x in self_first_5])}]")
    print(f"After self-attn block - Last 5:  [{', '.join([f'{x:.6f}' for x in self_last_5])}]")
    
    # SUBLAYER 2: Cross-Attention Block
    print(f"\n--- Layer {layer_idx} Cross-Attention Block ---")
    residual_cross = hidden_states_self
    
    # Pre-layer norm for cross-attention
    if layer.normalize_before:
        hidden_states_norm = layer.encoder_attn_layer_norm(hidden_states_self)
    else:
        hidden_states_norm = hidden_states_self
    
    cross_norm_first_5 = hidden_states_norm[0, 0, :5].cpu()
    cross_norm_last_5 = hidden_states_norm[0, 0, -5:].cpu()
    print(f"After cross-attn norm - First 5: [{', '.join([f'{x:.6f}' for x in cross_norm_first_5])}]")
    print(f"After cross-attn norm - Last 5:  [{', '.join([f'{x:.6f}' for x in cross_norm_last_5])}]")
    
    # Cross-attention
    hidden_states_cross, _, _ = layer.encoder_attn(
        hidden_states=hidden_states_norm,
        key_value_states=encoder_hidden_states,
        attention_mask=cross_attention_mask,
        layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
    )
    
    cross_attn_first_5 = hidden_states_cross[0, 0, :5].cpu()
    cross_attn_last_5 = hidden_states_cross[0, 0, -5:].cpu()
    print(f"After cross-attention - First 5: [{', '.join([f'{x:.6f}' for x in cross_attn_first_5])}]")
    print(f"After cross-attention - Last 5:  [{', '.join([f'{x:.6f}' for x in cross_attn_last_5])}]")
    
    # Residual connection
    hidden_states_cross = residual_cross + hidden_states_cross
    
    # Post-layer norm for cross-attention (if not normalize_before)
    if not layer.normalize_before:
        hidden_states_cross = layer.encoder_attn_layer_norm(hidden_states_cross)
    
    cross_first_5 = hidden_states_cross[0, 0, :5].cpu()
    cross_last_5 = hidden_states_cross[0, 0, -5:].cpu()
    print(f"After cross-attn block - First 5: [{', '.join([f'{x:.6f}' for x in cross_first_5])}]")
    print(f"After cross-attn block - Last 5:  [{', '.join([f'{x:.6f}' for x in cross_last_5])}]")
    
    # SUBLAYER 3: Feed-Forward Block
    print(f"\n--- Layer {layer_idx} Feed-Forward Block ---")
    residual_ffn = hidden_states_cross
    
    # Pre-layer norm for FFN
    if layer.normalize_before:
        hidden_states_norm = layer.final_layer_norm(hidden_states_cross)
    else:
        hidden_states_norm = hidden_states_cross
    
    ffn_norm_first_5 = hidden_states_norm[0, 0, :5].cpu()
    ffn_norm_last_5 = hidden_states_norm[0, 0, -5:].cpu()
    print(f"After FFN norm - First 5: [{', '.join([f'{x:.6f}' for x in ffn_norm_first_5])}]")
    print(f"After FFN norm - Last 5:  [{', '.join([f'{x:.6f}' for x in ffn_norm_last_5])}]")
    
    # FFN: FC1 -> activation -> FC2
    hidden_states_ffn = layer.activation_fn(layer.fc1(hidden_states_norm))
    
    fc1_first_5 = hidden_states_ffn[0, 0, :5].cpu()
    fc1_last_5 = hidden_states_ffn[0, 0, -5:].cpu()
    print(f"After FC1 + activation - First 5: [{', '.join([f'{x:.6f}' for x in fc1_first_5])}]")
    print(f"After FC1 + activation - Last 5:  [{', '.join([f'{x:.6f}' for x in fc1_last_5])}]")
    
    hidden_states_ffn = layer.fc2(hidden_states_ffn)
    
    fc2_first_5 = hidden_states_ffn[0, 0, :5].cpu()
    fc2_last_5 = hidden_states_ffn[0, 0, -5:].cpu()
    print(f"After FC2 - First 5: [{', '.join([f'{x:.6f}' for x in fc2_first_5])}]")
    print(f"After FC2 - Last 5:  [{', '.join([f'{x:.6f}' for x in fc2_last_5])}]")
    
    # Residual connection
    hidden_states_ffn = residual_ffn + hidden_states_ffn
    
    # Post-layer norm for FFN (if not normalize_before)
    if not layer.normalize_before:
        hidden_states_ffn = layer.final_layer_norm(hidden_states_ffn)
    
    ffn_first_5 = hidden_states_ffn[0, 0, :5].cpu()
    ffn_last_5 = hidden_states_ffn[0, 0, -5:].cpu()
    print(f"After FFN block - First 5: [{', '.join([f'{x:.6f}' for x in ffn_first_5])}]")
    print(f"After FFN block - Last 5:  [{', '.join([f'{x:.6f}' for x in ffn_last_5])}]")
    
    # Final layer 0 output
    current_hidden_states = hidden_states_ffn
    output_first_5 = current_hidden_states[0, 0, :5].cpu()
    output_last_5 = current_hidden_states[0, 0, -5:].cpu()
    print(f"\nLayer {layer_idx} FINAL Output - First 5: [{', '.join([f'{x:.6f}' for x in output_first_5])}]")
    print(f"Layer {layer_idx} FINAL Output - Last 5:  [{', '.join([f'{x:.6f}' for x in output_last_5])}]")
    
    # Process remaining layers normally (without detailed debug)
    for layer_idx in range(1, len(decoder.layers)):
        layer = decoder.layers[layer_idx]
        layer_outputs = layer(
            current_hidden_states,
            attention_mask=causal_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=cross_attention_mask,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        current_hidden_states = layer_outputs[0]
        output_first_5 = current_hidden_states[0, 0, :5].cpu()
        output_last_5 = current_hidden_states[0, 0, -5:].cpu()
        print(f"\nLayer {layer_idx} FINAL Output - First 5: [{', '.join([f'{x:.6f}' for x in output_first_5])}]")
        print(f"Layer {layer_idx} FINAL Output - Last 5:  [{', '.join([f'{x:.6f}' for x in output_last_5])}]")
    
    if hasattr(decoder, 'layer_norm') and decoder.layer_norm is not None:
        current_hidden_states = decoder.layer_norm(current_hidden_states)
    
    decoder_output = current_hidden_states

print(f"final_decoder_output_shape: {list(decoder_output.shape)}")
print(f"final_decoder_first_5: [{', '.join([f'{x:.6f}' for x in decoder_output[0, 0, :5].cpu()])}]")
print(f"final_decoder_last_5: [{', '.join([f'{x:.6f}' for x in decoder_output[0, 0, -5:].cpu()])}]")

print_task("logits_computation")
with torch.no_grad():
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(decoder_output)
    else:
        logits = torch.matmul(decoder_output, decoder.embed_tokens.weight.T)
    
    print(f"logits_shape: {list(logits.shape)}")
    print(f"logits_first_5: [{', '.join([f'{x:.6f}' for x in logits[0, 0, :5].cpu()])}]")
    print(f"logits_last_5: [{', '.join([f'{x:.6f}' for x in logits[0, 0, -5:].cpu()])}]")

    probabilities = torch.softmax(logits[0, 0], dim=0)
    next_token_id = torch.argmax(logits[0, 0]).item()
    next_token_prob = probabilities[next_token_id].item()
    
    print(f"next_token_id: {next_token_id}")
    print(f"next_token_prob: {next_token_prob:.6f}")
    
    if hasattr(tokenizer, 'decode'):
        try:
            next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(f"next_token_text: '{next_token_text}'")
        except:
            print(f"next_token_text: [decode_error]")

print_task("validation_with_generate")
with torch.no_grad():
    generated = model.generate(**inputs, max_length=50, num_beams=5)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translation = ip.postprocess_batch(decoded, lang=tgt_lang)
    print(f"full_translation: '{translation[0]}'")
    
    print(f"input_length: {len(inputs.input_ids[0])}")
    print(f"generated_length: {len(generated[0])}")
    print(f"input_tokens: {inputs.input_ids[0].tolist()}")
    print(f"generated_tokens: {generated[0].tolist()}")
    
    if len(generated[0]) > len(inputs.input_ids[0]):
        first_generated_token = generated[0, len(inputs.input_ids[0])].item()
        print(f"generated_first_token_id: {first_generated_token}")
        print(f"manual_vs_generated_match: {next_token_id == first_generated_token}")
    else:
        print("Generated sequence is not longer than input - no new tokens generated")
        
        decoder_start_tokens = model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') else model.config.bos_token_id
        if decoder_start_tokens is None:
            decoder_start_tokens = tokenizer.bos_token_id
        
        manual_generated = model.generate(
            **inputs, 
            decoder_input_ids=torch.tensor([[decoder_start_tokens]], device=device),
            max_length=len(inputs.input_ids[0]) + 2, 
            num_beams=1,
            do_sample=False
        )
        print(f"manual_generated_tokens: {manual_generated[0].tolist()}")
        if len(manual_generated[0]) > 1:
            manual_first_token = manual_generated[0, 1].item()
            print(f"manual_generated_first_token_id: {manual_first_token}")
            print(f"manual_vs_manual_match: {next_token_id == manual_first_token}")
        else:
            print("Even manual generation failed to produce tokens")

print_section("DECODER VALIDATION COMPLETE") 