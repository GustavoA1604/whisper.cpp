#!/usr/bin/env python3
"""
Simple test script for IndicTrans2 translation
Demonstrates English-to-Hindi and Hindi-to-English translation
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

def load_model(model_name):
    """Load the specified IndicTrans2 model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    return tokenizer, model, device

def translate_text(sentences, src_lang, tgt_lang, tokenizer, model, device):
    """Translate a list of sentences"""
    
    # Initialize the processor
    ip = IndicProcessor(inference=True)
    
    # Preprocess the input
    batch = ip.preprocess_batch(
        sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    
    # Tokenize
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    
    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    
    # Decode generated tokens
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    # Postprocess the translations
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    
    return translations

def main():
    """Main function to test translation"""
    
    print("="*60)
    print("IndicTrans2 Translation Test")
    print("="*60)
    
    # Test sentences
    english_sentences = [
        "Hello, how are you?",
        "This is a beautiful day.",
        "I love learning new languages.",
        "Machine translation is fascinating.",
        "Welcome to India!"
    ]
    
    hindi_sentences = [
        "नमस्ते, आप कैसे हैं?",
        "यह एक सुंदर दिन है।",
        "मुझे नई भाषाएं सीखना पसंद है।",
        "मशीन अनुवाद दिलचस्प है।",
        "भारत में आपका स्वागत है!"
    ]
    
    try:
        # Test English to Hindi translation
        print("\n1. Testing English to Hindi Translation:")
        print("-" * 40)
        
        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"  # Using distilled model for faster loading
        tokenizer, model, device = load_model(model_name)
        
        translations = translate_text(
            english_sentences, 
            "eng_Latn", 
            "hin_Deva", 
            tokenizer, 
            model, 
            device
        )
        
        for eng, hin in zip(english_sentences, translations):
            print(f"EN: {eng}")
            print(f"HI: {hin}")
            print()
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Test Hindi to English translation
        print("\n2. Testing Hindi to English Translation:")
        print("-" * 40)
        
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        tokenizer, model, device = load_model(model_name)
        
        translations = translate_text(
            hindi_sentences, 
            "hin_Deva", 
            "eng_Latn", 
            tokenizer, 
            model, 
            device
        )
        
        for hin, eng in zip(hindi_sentences, translations):
            print(f"HI: {hin}")
            print(f"EN: {eng}")
            print()
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nPossible solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check internet connection for model download")
        print("3. Ensure sufficient disk space for model files")

if __name__ == "__main__":
    main() 