#!/usr/bin/env python3
"""
Batch Translation Script for IndicTrans2
Translate text files from one language to another
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import argparse
import sys
from pathlib import Path

def load_model(model_name, device):
    """Load the specified model and tokenizer"""
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print(f"✓ Model loaded on {device}")
        return tokenizer, model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def get_model_name(src_lang, tgt_lang):
    """Get appropriate model name based on language pair"""
    if src_lang == 'eng_Latn':
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    elif tgt_lang == 'eng_Latn':
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    else:
        return "ai4bharat/indictrans2-indic-indic-dist-320M"

def translate_batch(sentences, src_lang, tgt_lang, tokenizer, model, processor, device, batch_size=8):
    """Translate a batch of sentences"""
    translations = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size}")
        
        # Preprocess
        processed_batch = processor.preprocess_batch(
            batch, src_lang=src_lang, tgt_lang=tgt_lang
        )
        
        # Tokenize
        inputs = tokenizer(
            processed_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode
        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Postprocess
        batch_translations = processor.postprocess_batch(decoded, lang=tgt_lang)
        translations.extend(batch_translations)
    
    return translations

def main():
    parser = argparse.ArgumentParser(description="Batch translate text files using IndicTrans2")
    parser.add_argument("input_file", help="Input text file to translate")
    parser.add_argument("output_file", help="Output file for translations")
    parser.add_argument("--src-lang", required=True, help="Source language code (e.g., eng_Latn)")
    parser.add_argument("--tgt-lang", required=True, help="Target language code (e.g., hin_Deva)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for translation")
    parser.add_argument("--model", help="Custom model name (optional)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Get model name
    if args.model:
        model_name = args.model
    else:
        model_name = get_model_name(args.src_lang, args.tgt_lang)
    
    print(f"Source language: {args.src_lang}")
    print(f"Target language: {args.tgt_lang}")
    print(f"Model: {model_name}")
    
    # Load model
    tokenizer, model = load_model(model_name, device)
    if tokenizer is None or model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Initialize processor
    processor = IndicProcessor(inference=True)
    
    # Read input file
    print(f"Reading input file: {args.input_file}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Found {len(lines)} lines to translate")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    if not lines:
        print("No text found in input file")
        sys.exit(1)
    
    # Translate
    print("Starting translation...")
    try:
        translations = translate_batch(
            lines, args.src_lang, args.tgt_lang, 
            tokenizer, model, processor, device, args.batch_size
        )
        
        # Write output
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        print(f"✓ Translation complete! Output written to: {args.output_file}")
        print(f"Translated {len(translations)} lines")
        
    except Exception as e:
        print(f"Translation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Show usage if no arguments
    if len(sys.argv) == 1:
        print("IndicTrans2 Batch Translation Tool")
        print("="*40)
        print("\nUsage:")
        print("  python batch_translate.py input.txt output.txt --src-lang eng_Latn --tgt-lang hin_Deva")
        print("\nCommon Language Codes:")
        print("  eng_Latn  - English")
        print("  hin_Deva  - Hindi")
        print("  ben_Beng  - Bengali")
        print("  tam_Taml  - Tamil")
        print("  tel_Telu  - Telugu")
        print("  guj_Gujr  - Gujarati")
        print("  mar_Deva  - Marathi")
        print("  kan_Knda  - Kannada")
        print("  mal_Mlym  - Malayalam")
        print("\nExample:")
        print("  python batch_translate.py english.txt hindi.txt --src-lang eng_Latn --tgt-lang hin_Deva")
        print("\nFor full language list, see README.md")
        sys.exit(0)
    
    main() 