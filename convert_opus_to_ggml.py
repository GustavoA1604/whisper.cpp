#!/usr/bin/env python3
"""
Convert Hugging Face Marian models to GGML format for whisper.cpp.
Simplified version that follows whisper.cpp's exact expectations.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import MarianConfig, MarianTokenizer
import sentencepiece as spm


def load_marian_model(model_path: Path):
    """Load Marian model from directory."""
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = MarianConfig.from_dict(config_dict)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    model_file = model_path / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    state_dict = torch.load(model_file, map_location='cpu')
    
    return config, tokenizer, state_dict


def write_header(fout, config, use_f16=True):
    """Write GGML header in exact whisper.cpp format."""
    # Magic number: "ggml" in hex
    fout.write(struct.pack("i", 0x67676d6c))
    
    fout.write(struct.pack("i", config.vocab_size))
    fout.write(struct.pack("i", config.max_position_embeddings))
    fout.write(struct.pack("i", config.d_model))
    fout.write(struct.pack("i", config.encoder_attention_heads))
    fout.write(struct.pack("i", config.encoder_layers))
    fout.write(struct.pack("i", config.max_position_embeddings))
    fout.write(struct.pack("i", config.d_model))
    fout.write(struct.pack("i", config.decoder_attention_heads))
    fout.write(struct.pack("i", config.decoder_layers))
    fout.write(struct.pack("i", 0))
    fout.write(struct.pack("i", 1 if use_f16 else 0))


def write_vocabulary(fout, tokenizer):
    """Write vocabulary in GGML format."""
    vocab = tokenizer.get_vocab()
    
    tokens = [""] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    
    fout.write(struct.pack("i", len(tokens)))
    
    for token in tokens:
        token_bytes = token.encode('utf-8')
        fout.write(struct.pack("i", len(token_bytes)))
        fout.write(token_bytes)


def write_tensor(fout, name: str, data: np.ndarray, use_f16: bool = True):
    """Write a single tensor in GGML format."""
    n_dims = len(data.shape)
    
    if use_f16 and n_dims >= 2 and "bias" not in name and "layer_norm" not in name and name != "encoder.embed_positions.weight":
        data = data.astype(np.float16)
        ftype = 1
        ftype_string = "fp16"
    else:
        data = data.astype(np.float32)
        ftype = 0
        ftype_string = "fp32"
    
    print(f"Writing tensor: {name} {list(data.shape)} (ftype={ftype_string})")
    
    name_bytes = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))
    
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    
    fout.write(name_bytes)
    
    data.tofile(fout)


def get_tensor_name_mapping():
    """Map Marian tensor names to whisper.cpp expected names."""
    return {
        # Embeddings
        "model.shared.weight": "encoder.embeddings.weight",
        "model.encoder.embed_tokens.weight": "encoder.embeddings.weight",
        "model.decoder.embed_tokens.weight": "decoder.embeddings.weight",
        "model.encoder.embed_positions.weight": "encoder.embed_positions.weight", 
        "model.decoder.embed_positions.weight": "decoder.embed_positions.weight",
        
        # Layer norms
        "model.encoder.layernorm_embedding.weight": "encoder.layer_norm.weight",
        "model.encoder.layernorm_embedding.bias": "encoder.layer_norm.bias",
        "model.decoder.layernorm_embedding.weight": "decoder.layer_norm.weight", 
        "model.decoder.layernorm_embedding.bias": "decoder.layer_norm.bias",
        
        # Final logits bias
        "final_logits_bias": "final_logits_bias",
        
        # Encoder layers
        "model.encoder.layers.{}.self_attn_layer_norm.weight": "encoder.layers.{}.self_attn_layer_norm.weight",
        "model.encoder.layers.{}.self_attn_layer_norm.bias": "encoder.layers.{}.self_attn_layer_norm.bias",
        "model.encoder.layers.{}.self_attn.q_proj.weight": "encoder.layers.{}.self_attn.q_proj.weight",
        "model.encoder.layers.{}.self_attn.q_proj.bias": "encoder.layers.{}.self_attn.q_proj.bias", 
        "model.encoder.layers.{}.self_attn.k_proj.weight": "encoder.layers.{}.self_attn.k_proj.weight",
        "model.encoder.layers.{}.self_attn.k_proj.bias": "encoder.layers.{}.self_attn.k_proj.bias",
        "model.encoder.layers.{}.self_attn.v_proj.weight": "encoder.layers.{}.self_attn.v_proj.weight",
        "model.encoder.layers.{}.self_attn.v_proj.bias": "encoder.layers.{}.self_attn.v_proj.bias",
        "model.encoder.layers.{}.self_attn.out_proj.weight": "encoder.layers.{}.self_attn.out_proj.weight",
        "model.encoder.layers.{}.self_attn.out_proj.bias": "encoder.layers.{}.self_attn.out_proj.bias",
        "model.encoder.layers.{}.final_layer_norm.weight": "encoder.layers.{}.final_layer_norm.weight",
        "model.encoder.layers.{}.final_layer_norm.bias": "encoder.layers.{}.final_layer_norm.bias",
        "model.encoder.layers.{}.fc1.weight": "encoder.layers.{}.fc1.weight",
        "model.encoder.layers.{}.fc1.bias": "encoder.layers.{}.fc1.bias",
        "model.encoder.layers.{}.fc2.weight": "encoder.layers.{}.fc2.weight",
        "model.encoder.layers.{}.fc2.bias": "encoder.layers.{}.fc2.bias",
        
        # Decoder layers
        "model.decoder.layers.{}.self_attn_layer_norm.weight": "decoder.blocks.{}.attn_ln.weight",
        "model.decoder.layers.{}.self_attn_layer_norm.bias": "decoder.blocks.{}.attn_ln.bias",
        "model.decoder.layers.{}.self_attn.q_proj.weight": "decoder.blocks.{}.attn.query.weight",
        "model.decoder.layers.{}.self_attn.q_proj.bias": "decoder.blocks.{}.attn.query.bias",
        "model.decoder.layers.{}.self_attn.k_proj.weight": "decoder.blocks.{}.attn.key.weight",
        "model.decoder.layers.{}.self_attn.k_proj.bias": "decoder.blocks.{}.attn.key.bias",
        "model.decoder.layers.{}.self_attn.v_proj.weight": "decoder.blocks.{}.attn.value.weight",
        "model.decoder.layers.{}.self_attn.v_proj.bias": "decoder.blocks.{}.attn.value.bias",
        "model.decoder.layers.{}.self_attn.out_proj.weight": "decoder.blocks.{}.attn.out.weight",
        "model.decoder.layers.{}.self_attn.out_proj.bias": "decoder.blocks.{}.attn.out.bias",
        
        # Cross-attention components
        "model.decoder.layers.{}.encoder_attn_layer_norm.weight": "decoder.blocks.{}.cross_attn_ln.weight",
        "model.decoder.layers.{}.encoder_attn_layer_norm.bias": "decoder.blocks.{}.cross_attn_ln.bias",
        "model.decoder.layers.{}.encoder_attn.q_proj.weight": "decoder.blocks.{}.cross_attn.query.weight",
        "model.decoder.layers.{}.encoder_attn.q_proj.bias": "decoder.blocks.{}.cross_attn.query.bias",
        "model.decoder.layers.{}.encoder_attn.k_proj.weight": "decoder.blocks.{}.cross_attn.key.weight",
        "model.decoder.layers.{}.encoder_attn.k_proj.bias": "decoder.blocks.{}.cross_attn.key.bias",
        "model.decoder.layers.{}.encoder_attn.v_proj.weight": "decoder.blocks.{}.cross_attn.value.weight",
        "model.decoder.layers.{}.encoder_attn.v_proj.bias": "decoder.blocks.{}.cross_attn.value.bias",
        "model.decoder.layers.{}.encoder_attn.out_proj.weight": "decoder.blocks.{}.cross_attn.out.weight",
        "model.decoder.layers.{}.encoder_attn.out_proj.bias": "decoder.blocks.{}.cross_attn.out.bias",
        
        # MLP components
        "model.decoder.layers.{}.final_layer_norm.weight": "decoder.blocks.{}.mlp_ln.weight",
        "model.decoder.layers.{}.final_layer_norm.bias": "decoder.blocks.{}.mlp_ln.bias",
        "model.decoder.layers.{}.fc1.weight": "decoder.blocks.{}.mlp.0.weight",
        "model.decoder.layers.{}.fc1.bias": "decoder.blocks.{}.mlp.0.bias",
        "model.decoder.layers.{}.fc2.weight": "decoder.blocks.{}.mlp.2.weight",
        "model.decoder.layers.{}.fc2.bias": "decoder.blocks.{}.mlp.2.bias",
    }


def convert_tensor_name(original_name: str) -> str:
    """Convert Marian tensor name to whisper.cpp expected name."""
    name_mapping = get_tensor_name_mapping()
    
    for pattern, replacement in name_mapping.items():
        if "{}" in pattern:
            import re
            pattern_regex = pattern.replace("{}", r"(\d+)")
            match = re.match(pattern_regex, original_name)
            if match:
                layer_num = match.group(1)
                return replacement.format(layer_num)
        elif pattern == original_name:
            return replacement
    
    return original_name


def convert_marian_to_ggml(model_path: Path, output_path: Path, use_f16: bool = True):
    """Convert Marian model to GGML format."""
    print(f"Loading Marian model from {model_path}")
    config, tokenizer, state_dict = load_marian_model(model_path)
    
    print(f"Model config: {config}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print(f"Total tensors: {len(state_dict)}")
    
    vocab = tokenizer.get_vocab()
    vocab_tokens = [""] * len(vocab)
    for token, idx in vocab.items():
        vocab_tokens[idx] = token
    
    print(f"Vocabulary size: {len(vocab_tokens)}")
    
    try:
        serialized_sp_model = tokenizer.spm_source.serialized_model_proto()
        print(f"Extracted original SentencePiece model: {len(serialized_sp_model)} bytes")
    except Exception as e:
        print(f"Warning: Could not extract SentencePiece model: {e}")
        print("Creating empty SentencePiece model data...")
        serialized_sp_model = b""
    
    with open(output_path, 'wb') as fout:
        print("Writing header...")
        write_header(fout, config, use_f16)
        
        print("Writing vocabulary...")
        write_vocabulary(fout, tokenizer)
        
        fout.write(struct.pack("i", len(serialized_sp_model)))
        if len(serialized_sp_model) > 0:
            fout.write(serialized_sp_model)
        
        print(f"Converting {len(state_dict)} tensors...")
        shared_tensor = None
        
        for name, tensor in state_dict.items():
            if name == "model.shared.weight":
                shared_tensor = tensor.clone()
            
            if name == "model.encoder.embed_tokens.weight" and "model.shared.weight" in state_dict:
                if torch.equal(tensor, state_dict["model.shared.weight"]):
                    print(f"Skipping duplicate tensor: {name} (same as shared)")
                    continue
            
            ggml_name = convert_tensor_name(name)
            
            print("Tensor : {0} is {1}".format(ggml_name,tensor.dtype))
            if tensor.dtype == torch.float16:
                np_tensor = tensor.to(torch.float32).numpy()
            else:
                np_tensor = tensor.numpy()
            
            write_tensor(fout, ggml_name, np_tensor, use_f16)
        
        if shared_tensor is not None:
            print("Adding lm_head.weight from shared embeddings")
            if shared_tensor.dtype == torch.float16:
                np_tensor = shared_tensor.to(torch.float32).numpy()
            else:
                np_tensor = shared_tensor.numpy()
            write_tensor(fout, "lm_head.weight", np_tensor, use_f16)
    
    print(f"Conversion complete! GGML file saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Marian model to GGML format")
    parser.add_argument("--model-dir", type=str, default="opus-mt-en-it",
                        help="Path to Marian model directory")
    parser.add_argument("--output", type=str, default="ggml-opus-en-it.bin",
                        help="Output GGML file path")
    parser.add_argument("--use-f32", action="store_true",
                        help="Use 32-bit floats instead of 16-bit")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_dir)
    output_path = Path(args.output)
    use_f16 = not args.use_f32
    
    if not model_path.exists():
        print(f"Error: Model directory {model_path} does not exist")
        return 1
    
    try:
        convert_marian_to_ggml(model_path, output_path, use_f16)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 