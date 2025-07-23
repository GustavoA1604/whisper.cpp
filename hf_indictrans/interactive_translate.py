#!/usr/bin/env python3
"""
Interactive IndicTrans2 Translation Tool
Allows real-time translation between supported languages
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import sys

class IndicTransTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.processor = IndicProcessor(inference=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Language mappings
        self.languages = {
            'english': 'eng_Latn',
            'hindi': 'hin_Deva', 
            'bengali': 'ben_Beng',
            'tamil': 'tam_Taml',
            'telugu': 'tel_Telu',
            'gujarati': 'guj_Gujr',
            'marathi': 'mar_Deva',
            'kannada': 'kan_Knda',
            'malayalam': 'mal_Mlym',
            'punjabi': 'pan_Guru',
            'odia': 'ory_Orya',
            'assamese': 'asm_Beng',
            'urdu': 'urd_Arab',
            'nepali': 'npi_Deva',
            'sanskrit': 'san_Deva',
            'konkani': 'gom_Deva',
            'manipuri': 'mni_Mtei',
            'bodo': 'brx_Deva',
            'dogri': 'doi_Deva',
            'kashmiri': 'kas_Deva',
            'maithili': 'mai_Deva',
            'santali': 'sat_Olck',
            'sindhi': 'snd_Deva'
        }
    
    def get_model_name(self, src_lang, tgt_lang):
        """Get appropriate model name based on language pair"""
        if src_lang == 'eng_Latn':
            return "ai4bharat/indictrans2-en-indic-dist-200M"
        elif tgt_lang == 'eng_Latn':
            return "ai4bharat/indictrans2-indic-en-dist-200M"
        else:
            return "ai4bharat/indictrans2-indic-indic-dist-320M"
    
    def load_model(self, model_name):
        """Load model and tokenizer if not already loaded"""
        if model_name not in self.models:
            print(f"Loading {model_name}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                model = model.to(self.device)
                
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                print("✓ Model loaded successfully!")
                
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                return False
        return True
    
    def translate(self, text, src_lang, tgt_lang):
        """Translate text from source to target language"""
        model_name = self.get_model_name(src_lang, tgt_lang)
        
        if not self.load_model(model_name):
            return None
        
        try:
            # Preprocess
            batch = self.processor.preprocess_batch(
                [text], src_lang=src_lang, tgt_lang=tgt_lang
            )
            
            # Tokenize
            inputs = self.tokenizers[model_name](
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_tokens = self.models[model_name].generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            # Decode
            generated_tokens = self.tokenizers[model_name].batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Postprocess
            translations = self.processor.postprocess_batch(generated_tokens, lang=tgt_lang)
            return translations[0] if translations else None
            
        except Exception as e:
            print(f"Translation error: {e}")
            return None
    
    def show_languages(self):
        """Display supported languages"""
        print("\nSupported Languages:")
        print("=" * 40)
        for name, code in sorted(self.languages.items()):
            print(f"{name.capitalize():15} -> {code}")
        print()
    
    def interactive_mode(self):
        """Run interactive translation mode"""
        print("="*60)
        print("IndicTrans2 Interactive Translation Tool")
        print("="*60)
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                print("\n" + "-"*40)
                command = input("Enter command (translate/help/languages/quit): ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif command in ['help', 'h']:
                    self.show_help()
                
                elif command in ['languages', 'lang', 'l']:
                    self.show_languages()
                
                elif command in ['translate', 't', '']:
                    self.translation_session()
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def translation_session(self):
        """Handle translation session"""
        print("\nTranslation Session")
        print("Type 'back' to return to main menu")
        
        # Get source language
        while True:
            src_input = input("\nSource language (e.g., english, hindi): ").strip().lower()
            if src_input == 'back':
                return
            if src_input in self.languages:
                src_lang = self.languages[src_input]
                break
            else:
                print(f"Unknown language: {src_input}")
                print("Available languages:", ', '.join(self.languages.keys()))
        
        # Get target language
        while True:
            tgt_input = input("Target language (e.g., hindi, english): ").strip().lower()
            if tgt_input == 'back':
                return
            if tgt_input in self.languages:
                tgt_lang = self.languages[tgt_input]
                break
            else:
                print(f"Unknown language: {tgt_input}")
                print("Available languages:", ', '.join(self.languages.keys()))
        
        print(f"\nTranslating from {src_input.capitalize()} to {tgt_input.capitalize()}")
        print("Enter text to translate (or 'back' to return):")
        
        # Translation loop
        while True:
            text = input(f"\n{src_input.capitalize()}: ").strip()
            
            if text.lower() == 'back':
                break
            
            if not text:
                continue
            
            print("Translating...")
            translation = self.translate(text, src_lang, tgt_lang)
            
            if translation:
                print(f"{tgt_input.capitalize()}: {translation}")
            else:
                print("Translation failed. Please try again.")
    
    def show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("=" * 40)
        print("translate (t)  - Start translation session")
        print("languages (l) - Show supported languages")
        print("help (h)      - Show this help")
        print("quit (q)      - Exit the program")
        print("\nDuring translation:")
        print("- Enter text to translate")
        print("- Type 'back' to return to main menu")
        print("- Press Ctrl+C to exit anytime")

def main():
    try:
        translator = IndicTransTranslator()
        translator.interactive_mode()
    except Exception as e:
        print(f"Startup error: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Run the setup script: python setup.py")

if __name__ == "__main__":
    main() 