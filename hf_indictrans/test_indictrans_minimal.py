#!/usr/bin/env python3

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Load model and tokenizer
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Input sentence
sentence = "Hello world"

# Preprocess
ip = IndicProcessor(inference=True)
processed = ip.preprocess_batch([sentence], src_lang="eng_Latn", tgt_lang="hin_Deva")

# Tokenize
inputs = tokenizer(processed, return_tensors="pt", padding=True).to(device)
print("Tokenized inputs:", inputs.input_ids)
print("Input tokens decoded:", tokenizer.batch_decode(inputs.input_ids))

# Encoder forward pass
with torch.no_grad():
    encoder_outputs = model.get_encoder()(inputs.input_ids, attention_mask=inputs.attention_mask)
    print("Encoder output shape:", encoder_outputs.last_hidden_state.shape)
    print("Encoder output sample:", encoder_outputs.last_hidden_state[0, 0, :5])

# Generate translation
with torch.no_grad():
    generated_tokens = model.generate(**inputs, max_length=50, num_beams=5)

print("Generated tokens:", generated_tokens)
print("Generated tokens decoded:", tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# Postprocess
translation = ip.postprocess_batch(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True), lang="hin_Deva")
print("Final translation:", translation[0]) 