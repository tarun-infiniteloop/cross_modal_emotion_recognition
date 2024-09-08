# # pip install sentencepiece
# # pip install transformers pandas googletrans==4.0.0-rc1

# from transformers import MarianMTModel, MarianTokenizer
# import pandas as pd
# from tqdm import tqdm

# def load_model_and_tokenizer(src_lang="en", tgt_lang="fr"):
#     model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
#     model = MarianMTModel.from_pretrained(model_name)
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     return model, tokenizer

# def translate(text, model, tokenizer, return_text=True):
#     # Prepare the text for translation
#     batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
#     translated = model.generate(**batch)
#     tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#     return tgt_text[0] if return_text else tgt_text

# def translate_text(model_en_fr, tokenizer_en_fr, model_fr_en, tokenizer_fr_en, text):
#     # Translate to French
#     text_to_fr = translate(text, model_en_fr, tokenizer_en_fr)
#     # Translate back to English
#     text_to_en = translate(text_to_fr, model_fr_en, tokenizer_fr_en)
#     return text_to_en

# def augment_sentences(file_path):
#     df = pd.read_csv(file_path)
    
#     # Load models and tokenizers
#     model_en_fr, tokenizer_en_fr = load_model_and_tokenizer("en", "fr")
#     model_fr_en, tokenizer_fr_en = load_model_and_tokenizer("fr", "en")
    
#     results = []
#     for text in tqdm(df['Sentences'], desc="Translating Sentences"):
#         augmented_text = translate_text(model_en_fr, tokenizer_en_fr, model_fr_en, tokenizer_fr_en, text)
#         results.append(augmented_text)
    
#     df['augSentence'] = results
#     df.to_csv('./aug_balanced_text_label_file_iemocap.csv', index=False)

# # Example usage
# augment_sentences('/home/taruns/project/clip/new_codes_balanced_datasets/csv_file_generation_code_csv/balanced_text_label_file_iemocap.csv')


from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import torch

def load_model_and_tokenizer(src_lang="en", tgt_lang="fr", device='cuda'):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate(text, model, tokenizer, device='cuda', return_text=True):
    # Prepare the text for translation
    batch = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0] if return_text else tgt_text

def translate_text(model_en_fr, tokenizer_en_fr, model_fr_de, tokenizer_fr_de, model_de_en, tokenizer_de_en, text, device):
    # Translate to French
    text_to_fr = translate(text, model_en_fr, tokenizer_en_fr, device)
    # Translate to German
    text_to_de = translate(text_to_fr, model_fr_de, tokenizer_fr_de, device)
    # Translate back to English
    text_to_en = translate(text_to_de, model_de_en, tokenizer_de_en, device)
    return text_to_en

def augment_sentences(file_path, device='cuda'):
    df = pd.read_csv(file_path)
    
    # Check if CUDA is available and use it if possible
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models and tokenizers for each language transition
    model_en_fr, tokenizer_en_fr = load_model_and_tokenizer("en", "fr", device)
    model_fr_de, tokenizer_fr_de = load_model_and_tokenizer("fr", "de", device)
    model_de_en, tokenizer_de_en = load_model_and_tokenizer("de", "en", device)
    
    results = []
    for text in tqdm(df['Sentences'], desc="Translating Sentences"):
        augmented_text = translate_text(model_en_fr, tokenizer_en_fr, model_fr_de, tokenizer_fr_de, model_de_en, tokenizer_de_en, text, device)
        results.append(augmented_text)
    
    df['augSentence'] = results
    df.to_csv('./aug2_multi_lang_translated_text_label_file_iemocap.csv', index=False)

# Example usage
augment_sentences('/home/taruns/project/clip/new_codes_balanced_datasets/csv_file_generation_code_csv/balanced_text_label_file_iemocap.csv')
