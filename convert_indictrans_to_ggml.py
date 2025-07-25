#!/usr/bin/env python3
"""
Convert Hugging Face IndicTrans2 models to GGML format for whisper.cpp.
Based on convert_opus_to_ggml.py but adapted for IndicTrans2 models.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig


def extract_sentencepiece_model(tokenizer):
    """
    Comprehensive SentencePiece model extraction for IndicTrans2 tokenizer.
    IndicTrans2 has separate source and target SentencePiece models.
    """
    serialized_sp_model = b""
    extraction_method = "none"
    
    try:
        # Method 1: IndicTrans2 specific - extract source SentencePiece model
        if hasattr(tokenizer, 'src_spm') and tokenizer.src_spm is not None:
            serialized_sp_model = tokenizer.src_spm.serialized_model_proto()
            extraction_method = "indictrans_src_spm"
            print(f"✓ Extracted IndicTrans2 source SentencePiece model: {len(serialized_sp_model)} bytes")
            
        # Method 2: Try reading source SentencePiece model file directly
        elif hasattr(tokenizer, 'src_spm_fp') and tokenizer.src_spm_fp:
            import os
            if os.path.exists(tokenizer.src_spm_fp):
                with open(tokenizer.src_spm_fp, 'rb') as f:
                    serialized_sp_model = f.read()
                extraction_method = f"indictrans_src_file: {tokenizer.src_spm_fp}"
                print(f"✓ Extracted from IndicTrans2 source file: {len(serialized_sp_model)} bytes")
                
        # Method 3: Fallback to target SentencePiece model if source not available
        elif hasattr(tokenizer, 'tgt_spm') and tokenizer.tgt_spm is not None:
            serialized_sp_model = tokenizer.tgt_spm.serialized_model_proto()
            extraction_method = "indictrans_tgt_spm"
            print(f"✓ Extracted IndicTrans2 target SentencePiece model: {len(serialized_sp_model)} bytes")
            
        # Method 4: Try reading target SentencePiece model file directly
        elif hasattr(tokenizer, 'tgt_spm_fp') and tokenizer.tgt_spm_fp:
            import os
            if os.path.exists(tokenizer.tgt_spm_fp):
                with open(tokenizer.tgt_spm_fp, 'rb') as f:
                    serialized_sp_model = f.read()
                extraction_method = f"indictrans_tgt_file: {tokenizer.tgt_spm_fp}"
                print(f"✓ Extracted from IndicTrans2 target file: {len(serialized_sp_model)} bytes")
                
        # Method 5: Generic sp_model access (for other tokenizers)
        elif hasattr(tokenizer, 'sp_model') and tokenizer.sp_model is not None:
            serialized_sp_model = tokenizer.sp_model.serialized_model_proto()
            extraction_method = "generic_sp_model"
            print(f"✓ Extracted via generic sp_model: {len(serialized_sp_model)} bytes")
            
        # Method 6: Try to find .model file in tokenizer directory
        elif hasattr(tokenizer, 'name_or_path'):
            import os
            import glob
            
            # Look for .model files in the tokenizer directory
            base_path = tokenizer.name_or_path if os.path.isdir(tokenizer.name_or_path) else ""
            if base_path:
                model_files = glob.glob(os.path.join(base_path, "*.model"))
                if model_files:
                    model_file = model_files[0]  # Take the first .model file
                    with open(model_file, 'rb') as f:
                        serialized_sp_model = f.read()
                    extraction_method = f"model_file: {model_file}"
                    print(f"✓ Extracted from {model_file}: {len(serialized_sp_model)} bytes")
        
        # If we still don't have the model, provide detailed info
        if len(serialized_sp_model) == 0:
            print("⚠ Could not extract SentencePiece model")
            print("IndicTrans2 tokenizer analysis:")
            if hasattr(tokenizer, 'src_spm'):
                print(f"  - Has src_spm: {tokenizer.src_spm is not None}")
            if hasattr(tokenizer, 'tgt_spm'):
                print(f"  - Has tgt_spm: {tokenizer.tgt_spm is not None}")
            if hasattr(tokenizer, 'src_spm_fp'):
                print(f"  - Source model file: {tokenizer.src_spm_fp}")
            if hasattr(tokenizer, 'tgt_spm_fp'):
                print(f"  - Target model file: {tokenizer.tgt_spm_fp}")
                        
        return serialized_sp_model, extraction_method
        
    except Exception as e:
        print(f"Error in SentencePiece extraction: {e}")
        import traceback
        traceback.print_exc()
        return b"", "error"


def load_indictrans_model(model_path: Path):
    """Load IndicTrans2 model from directory or HuggingFace model name."""
    print(f"Loading IndicTrans2 model from {model_path}")
    
    # Convert Path to string for HuggingFace compatibility
    model_path_str = str(model_path)
    
    # Load config
    config = AutoConfig.from_pretrained(model_path_str, trust_remote_code=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path_str, trust_remote_code=True)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path_str, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # Get state dict
    state_dict = model.state_dict()
    
    return config, tokenizer, state_dict


def write_header(fout, config, use_f16=True):
    """Write GGML header adapted for IndicTrans2 format."""
    # Magic number: "ggml" in hex
    fout.write(struct.pack("i", 0x67676d6c))
    
    # IndicTrans2 specific configuration
    encoder_vocab_size = getattr(config, 'encoder_vocab_size', config.vocab_size)
    decoder_vocab_size = getattr(config, 'decoder_vocab_size', config.vocab_size)
    
    # Use same header format as Marian for consistency
    fout.write(struct.pack("i", encoder_vocab_size))  # n_vocab (encoder vocab size)
    fout.write(struct.pack("i", config.max_source_positions))  # n_audio_ctx (encoder max positions)
    fout.write(struct.pack("i", config.encoder_embed_dim))  # n_audio_state (encoder embedding dim)
    fout.write(struct.pack("i", config.encoder_attention_heads))  # n_audio_head
    fout.write(struct.pack("i", config.encoder_layers))  # n_audio_layer
    fout.write(struct.pack("i", config.max_target_positions))  # n_text_ctx (decoder max positions)
    fout.write(struct.pack("i", config.decoder_embed_dim))  # n_text_state (decoder embedding dim)
    fout.write(struct.pack("i", config.decoder_attention_heads))  # n_text_head
    fout.write(struct.pack("i", config.decoder_layers))  # n_text_layer
    fout.write(struct.pack("i", 2))  # Model type: 2 = INDICTRANS (field_10)
    fout.write(struct.pack("i", 1 if use_f16 else 0))  # use f16
    fout.write(struct.pack("i", decoder_vocab_size))  # decoder vocab size (field_12 - IndicTrans2 specific)


def write_vocabulary(fout, tokenizer):
    """Write vocabulary in GGML format for IndicTrans2."""
    # Get vocabulary from tokenizer
    vocab = tokenizer.get_vocab()
    
    # Create ordered token list
    tokens = [""] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    
    print(f"Writing vocabulary with {len(tokens)} tokens")
    fout.write(struct.pack("i", len(tokens)))
    
    for token in tokens:
        token_bytes = token.encode('utf-8')
        fout.write(struct.pack("i", len(token_bytes)))
        fout.write(token_bytes)


def write_tensor(fout, name: str, data: np.ndarray, use_f16: bool = True):
    """Write a single tensor in GGML format."""
    n_dims = len(data.shape)
    
    # Determine precision based on tensor characteristics
    if use_f16 and n_dims >= 2 and "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
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
    
    # Write dimensions in reverse order (GGML convention)
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    
    fout.write(name_bytes)
    data.tofile(fout)


def get_indictrans_tensor_name_mapping():
    """Map IndicTrans2 tensor names to whisper.cpp expected names."""
    return {
        # Embeddings
        "model.shared.weight": "encoder.embeddings.weight",
        "model.encoder.embed_tokens.weight": "encoder.embeddings.weight",
        "model.decoder.embed_tokens.weight": "decoder.embeddings.weight",
        # Note: IndicTrans2 uses sinusoidal positional embeddings (no learnable weights)
        # "model.encoder.embed_positions.weight": "encoder.embed_positions.weight",  # Not present in IndicTrans2
        # "model.decoder.embed_positions.weight": "decoder.embed_positions.weight",  # Not present in IndicTrans2
        
        # Layer normalization for embeddings
        "model.encoder.layernorm_embedding.weight": "encoder.layer_norm.weight",
        "model.encoder.layernorm_embedding.bias": "encoder.layer_norm.bias",
        "model.decoder.layernorm_embedding.weight": "decoder.layer_norm.weight",
        "model.decoder.layernorm_embedding.bias": "decoder.layer_norm.bias",
        
        # Final layer norm
        "model.encoder.layer_norm.weight": "encoder.final_layer_norm.weight",
        "model.encoder.layer_norm.bias": "encoder.final_layer_norm.bias",
        "model.decoder.layer_norm.weight": "decoder.final_layer_norm.weight",
        "model.decoder.layer_norm.bias": "decoder.final_layer_norm.bias",
        
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
        
        # Decoder layers - self attention
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
        
        # Decoder layers - cross attention
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
        
        # Decoder layers - MLP
        "model.decoder.layers.{}.final_layer_norm.weight": "decoder.blocks.{}.mlp_ln.weight",
        "model.decoder.layers.{}.final_layer_norm.bias": "decoder.blocks.{}.mlp_ln.bias",
        "model.decoder.layers.{}.fc1.weight": "decoder.blocks.{}.mlp.0.weight",
        "model.decoder.layers.{}.fc1.bias": "decoder.blocks.{}.mlp.0.bias",
        "model.decoder.layers.{}.fc2.weight": "decoder.blocks.{}.mlp.2.weight",
        "model.decoder.layers.{}.fc2.bias": "decoder.blocks.{}.mlp.2.bias",
        
        # IndicTrans2 uses shared embeddings - no separate lm_head.weight
    }


def convert_tensor_name(original_name: str) -> str:
    """Convert IndicTrans2 tensor name to whisper.cpp expected name."""
    name_mapping = get_indictrans_tensor_name_mapping()
    
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
    
    # If no mapping found, return original name
    return original_name


def convert_indictrans_to_ggml(model_path: Path, output_path: Path, use_f16: bool = True):
    """Convert IndicTrans2 model to GGML format."""
    print(f"Loading IndicTrans2 model from {model_path}")
    config, tokenizer, state_dict = load_indictrans_model(model_path)
    
    print(f"Model config: {config}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print(f"Total tensors: {len(state_dict)}")
    
    # Get vocabulary information
    vocab = tokenizer.get_vocab()
    vocab_tokens = [""] * len(vocab)
    for token, idx in vocab.items():
        vocab_tokens[idx] = token
    
    print(f"Vocabulary size: {len(vocab_tokens)}")
    
    # Extract SentencePiece model from IndicTrans2 tokenizer
    print("Extracting SentencePiece model from IndicTrans2 tokenizer...")
    serialized_sp_model, extraction_method = extract_sentencepiece_model(tokenizer)
    
    if len(serialized_sp_model) > 0:
        print(f"✓ SentencePiece model successfully extracted using method: {extraction_method}")
    else:
        print("⚠ Warning: Could not extract SentencePiece model")
        print("This may cause tokenization issues in the converted model")
        print("IndicTrans2 tokenizer details:")
        print(f"  - Type: {type(tokenizer)}")
        print(f"  - Name/Path: {getattr(tokenizer, 'name_or_path', 'Unknown')}")
        print(f"  - Vocab size: {len(vocab)}")
        
        # Additional debugging info
        special_tokens = {}
        for attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                if token is not None:
                    special_tokens[attr] = str(token)
        
        if special_tokens:
            print(f"  - Special tokens: {special_tokens}")
    
    print(f"Final SentencePiece model size: {len(serialized_sp_model)} bytes")
    
    with open(output_path, 'wb') as fout:
        print("Writing header...")
        write_header(fout, config, use_f16)
        
        print("Writing vocabulary...")
        write_vocabulary(fout, tokenizer)
        
        # Write SentencePiece model data
        fout.write(struct.pack("i", len(serialized_sp_model)))
        if len(serialized_sp_model) > 0:
            fout.write(serialized_sp_model)
        
        print(f"Converting {len(state_dict)} tensors...")
        shared_tensor = None
        
        for name, tensor in state_dict.items():
            # Handle shared embeddings
            if name == "model.shared.weight":
                shared_tensor = tensor.clone()
            
            # Skip duplicate embeddings if they exist
            if name == "model.encoder.embed_tokens.weight" and "model.shared.weight" in state_dict:
                if torch.equal(tensor, state_dict["model.shared.weight"]):
                    print(f"Skipping duplicate tensor: {name} (same as shared)")
                    continue
            
            # Skip lm_head tensors for IndicTrans2 (uses shared embeddings)
            if "lm_head" in name:
                print(f"Skipping IndicTrans2 lm_head tensor: {name} (uses shared embeddings)")
                continue
            
            # Convert tensor name
            ggml_name = convert_tensor_name(name)
            
            print(f"Tensor: {ggml_name} is {tensor.dtype}")
            
            # Convert to numpy
            if tensor.dtype == torch.float16:
                np_tensor = tensor.to(torch.float32).numpy()
            else:
                np_tensor = tensor.numpy()
            
            write_tensor(fout, ggml_name, np_tensor, use_f16)
        
        # IndicTrans2 uses shared embeddings, no separate lm_head.weight needed
        if shared_tensor is not None:
            print("IndicTrans2 uses shared embeddings (model.shared.weight) - no separate lm_head needed")
    
    print(f"Conversion complete! GGML file saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert IndicTrans2 model to GGML format")
    parser.add_argument("--model-name", type=str, default="ai4bharat/indictrans2-en-indic-dist-200M",
                        help="HuggingFace model name or path to local model directory")
    parser.add_argument("--output", type=str, default="ggml-indictrans2-en-indic.bin",
                        help="Output GGML file path")
    parser.add_argument("--use-f32", action="store_true",
                        help="Use 32-bit floats instead of 16-bit")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_name)
    output_path = Path(args.output)
    use_f16 = not args.use_f32
    
    # Check if it's a local path or HuggingFace model name
    if model_path.exists() and model_path.is_dir():
        print(f"Using local model directory: {model_path}")
    else:
        print(f"Using HuggingFace model: {args.model_name}")
        model_path = args.model_name
    
    try:
        convert_indictrans_to_ggml(model_path, output_path, use_f16)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 