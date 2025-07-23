# IndicTrans2 Translation Setup

This folder contains a complete setup for using **IndicTrans2**, a state-of-the-art multilingual neural machine translation model that supports translation between English and 22 scheduled Indian languages.

## 🌟 Features

- **High-quality translation** between English and Indian languages
- **Inter-Indian language translation** (Hindi ↔ Bengali, etc.)
- Support for **22 Indian languages** in multiple scripts
- **Pre-trained models** from AI4Bharat/IIT Madras
- **Easy-to-use Python interface**

## 📋 Supported Languages

IndicTrans2 supports the following languages:

| Language | Code | Script |
|----------|------|--------|
| Assamese | `asm_Beng` | Bengali |
| Bengali | `ben_Beng` | Bengali |
| Bodo | `brx_Deva` | Devanagari |
| Dogri | `doi_Deva` | Devanagari |
| English | `eng_Latn` | Latin |
| Gujarati | `guj_Gujr` | Gujarati |
| Hindi | `hin_Deva` | Devanagari |
| Kannada | `kan_Knda` | Kannada |
| Kashmiri (Arabic) | `kas_Arab` | Arabic |
| Kashmiri (Devanagari) | `kas_Deva` | Devanagari |
| Konkani | `gom_Deva` | Devanagari |
| Maithili | `mai_Deva` | Devanagari |
| Malayalam | `mal_Mlym` | Malayalam |
| Manipuri (Bengali) | `mni_Beng` | Bengali |
| Manipuri (Meitei) | `mni_Mtei` | Meitei |
| Marathi | `mar_Deva` | Devanagari |
| Nepali | `npi_Deva` | Devanagari |
| Odia | `ory_Orya` | Odia |
| Punjabi | `pan_Guru` | Gurmukhi |
| Sanskrit | `san_Deva` | Devanagari |
| Santali | `sat_Olck` | Ol Chiki |
| Sindhi (Arabic) | `snd_Arab` | Arabic |
| Sindhi (Devanagari) | `snd_Deva` | Devanagari |
| Tamil | `tam_Taml` | Tamil |
| Telugu | `tel_Telu` | Telugu |
| Urdu | `urd_Arab` | Arabic |

## 🚀 Quick Start

### 1. Setup Environment

Run the setup script to install dependencies and download models:

```bash
python setup.py
```

This will:
- Install all required Python packages
- Download the IndicTrans2 models (distilled versions for faster loading)
- Verify the installation

### 2. Run the Test

After setup, test the installation:

```bash
python test_indictrans.py
```

This will demonstrate English ↔ Hindi translation with sample sentences.

## 📖 Manual Installation

If you prefer manual installation:

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers
- `IndicTransToolkit` - Preprocessing/postprocessing toolkit
- `sentencepiece` - Tokenization
- `huggingface_hub` - Model downloading

## 🔧 Usage Examples

### Basic Translation

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Load model
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# Initialize processor
ip = IndicProcessor(inference=True)

# Translate
sentences = ["Hello, how are you?", "This is a test."]
src_lang, tgt_lang = "eng_Latn", "hin_Deva"

# Preprocess
batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# Tokenize
inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, num_beams=5, max_length=256)

# Decode and postprocess
translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
final_translations = ip.postprocess_batch(translations, lang=tgt_lang)

print(final_translations)
```

### Advanced Usage

```python
# Custom translation function
def translate(text_list, src_lang, tgt_lang, model_name=None):
    if model_name is None:
        if src_lang == "eng_Latn":
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        else:
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)
    
    batch = ip.preprocess_batch(text_list, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=5, max_length=256)
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt_lang)

# Use the function
hindi_text = translate(["Hello world"], "eng_Latn", "hin_Deva")
print(hindi_text)  # ['हैलो वर्ल्ड']
```

## 🎯 Available Models

The setup uses distilled models for faster loading and inference:

| Model | Size | Direction | Use Case |
|-------|------|-----------|----------|
| `indictrans2-en-indic-dist-200M` | 200M | EN → Indic | English to Indian languages |
| `indictrans2-indic-en-dist-200M` | 200M | Indic → EN | Indian languages to English |
| `indictrans2-indic-indic-dist-320M` | 320M | Indic → Indic | Between Indian languages |

For higher quality (but slower), you can use the full models:
- `indictrans2-en-indic-1B` (1B parameters)
- `indictrans2-indic-en-1B` (1B parameters)

## 💡 Tips for Better Results

1. **Use proper language codes**: Always use the correct language codes (e.g., `hin_Deva` for Hindi)
2. **Preprocessing matters**: The `IndicProcessor` handles important preprocessing steps
3. **Batch processing**: Process multiple sentences together for efficiency
4. **GPU acceleration**: Use CUDA if available for faster inference
5. **Beam search**: Use `num_beams=5` for better quality (slower)

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Download Fails**: Check internet connection and disk space
   ```bash
   # Models are ~800MB each and cached in ~/.cache/huggingface/
   ```

3. **CUDA Out of Memory**: Use CPU or reduce batch size
   ```python
   model = model.to("cpu")  # Force CPU usage
   ```

4. **Slow Performance**: Use distilled models or enable GPU
   ```python
   model = AutoModelForSeq2SeqLM.from_pretrained(
       model_name, 
       trust_remote_code=True,
       torch_dtype=torch.float16  # Use half precision
   ).to("cuda")
   ```

### Performance Optimization

- **Use GPU**: Move models to CUDA for 5-10x speedup
- **Half precision**: Use `torch.float16` to reduce memory usage
- **Batch processing**: Process multiple sentences together
- **Model caching**: Models are cached after first download

## 📚 Additional Resources

- **Paper**: [IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages](https://openreview.net/forum?id=vfT4YuzAYA)
- **GitHub**: [AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
- **Hugging Face**: [AI4Bharat Organization](https://huggingface.co/ai4bharat)
- **IndicTransToolkit**: [GitHub Repository](https://github.com/VarunGumma/IndicTransToolkit)

## 📝 License

The models are released under MIT License. Please cite the original paper if you use IndicTrans2 in your research:

```bibtex
@article{gala2023indictrans,
  title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
  journal={Transactions on Machine Learning Research},
  year={2023},
  url={https://openreview.net/forum?id=vfT4YuzAYA}
}
```

## 🆘 Support

If you encounter issues:
1. Check this README for common solutions
2. Review the error messages carefully
3. Ensure all dependencies are correctly installed
4. Check the [IndicTrans2 GitHub repository](https://github.com/AI4Bharat/IndicTrans2) for updates

---

**Happy Translating! 🚀** 