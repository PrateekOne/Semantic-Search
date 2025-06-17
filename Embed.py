import pandas as pd
import os
from tkinter import Tk, filedialog
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from langdetect import detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from googletrans import Translator

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load NLLB model with fallback
try:
    hinglish_translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    hinglish_translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
    print("NLLB-200-distilled-600M model loaded for translation.")
    use_nllb = True
except Exception as e:
    print(f"Error loading NLLB translation model: {e}. Attempting to use googletrans as a fallback.")
    try:
        google_translator = Translator()
        print("googletrans Translator initialized as fallback.")
        use_nllb = False
    except Exception as ge:
        print(f"Error initializing googletrans: {ge}. No translation model available.")
        use_nllb = None

# Step 1: Select file
def select_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("CSV and JSON files", "*.csv *.json")])

# Step 2: Load dataset
def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, on_bad_lines='skip'), 'csv'
    elif file_path.endswith('.json'):
        return pd.read_json(file_path), 'json'
    else:
        raise ValueError("Unsupported file format")

# Step 3: Dataset type detection
def detect_dataset_type(columns):
    keywords = {
        'customer': ['name', 'email', 'customer', 'user'],
        'product': ['price', 'product', 'item', 'brand'],
        'location': ['latitude', 'longitude', 'location', 'city'],
        'review': ['review', 'rating', 'feedback', 'comment', 'description']
    }
    detected = set()
    for col in columns:
        for key, terms in keywords.items():
            if any(term in col.lower() for term in terms):
                detected.add(key)
    return list(detected)

# Step 4: Clean
def clean_dataframe(df):
    return df.astype(str).applymap(lambda x: x.strip().replace('\n', ' '))

# Step 5: Translate only non-English descriptions
def translate_descriptions(texts, batch_size=16):
    translated = []
    buffer = []

    for text in tqdm(texts, desc="🔄 Translating Descriptions"):
        try:
            if detect(text) != 'en':
                buffer.append(text)
            else:
                translated.append(text)
                continue
        except:
            translated.append(text)
            continue

        if len(buffer) == batch_size:
            translated_batch = translate_batch(buffer)
            translated.extend(translated_batch)
            buffer = []

    if buffer:
        translated_batch = translate_batch(buffer)
        translated.extend(translated_batch)

    return translated

def translate_batch(texts):
    if use_nllb:
        try:
            tokens = hinglish_translator_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            translated = hinglish_translator_model.generate(**tokens, max_length=512)
            return hinglish_translator_tokenizer.batch_decode(translated, skip_special_tokens=True)
        except Exception as e:
            print(f"NLLB translation failed: {e}")
            return texts
    elif use_nllb is False:
        try:
            return [google_translator.translate(text, dest='en').text for text in texts]
        except Exception as ge:
            print(f"Google translation failed: {ge}")
            return texts
    else:
        return texts

# Step 6: Embed rows
def embed_rows(df, num_rows, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    df_subset = df.iloc[:num_rows].copy()

    if 'description' in df.columns:
        df_subset['description'] = translate_descriptions(df_subset['description'].tolist())

    def row_to_text(row):
        return ' | '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and str(row[col]).strip() != ''])

    texts = df_subset.apply(row_to_text, axis=1).tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embeddings, texts, df_subset.index.tolist()

# Step 7: Save to JSON
def save_embeddings(embeddings, df, file_path, texts, indices, language="en"):
    out_data = [{
        "pointer": indices[i],
        "source_file": os.path.abspath(file_path),
        "summary": texts[i],
        "language": language,
        "embedding": emb.tolist()
    } for i, emb in enumerate(embeddings)]

    out_path = os.path.splitext(os.path.basename(file_path))[0] + '_processed_embeddings.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False)
    print(f"\n✅ Embeddings saved to {out_path}")

# Main
def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return
    
    df, _ = load_dataset(file_path)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print("Detected type:", detect_dataset_type(df.columns))

    df = clean_dataframe(df)
    max_rows = len(df)
    num_rows = int(input(f"\nHow many rows to embed (max {max_rows}): "))
    num_rows = min(num_rows, max_rows)

    embeddings, texts, indices = embed_rows(df, num_rows)
    save_embeddings(embeddings, df, file_path, texts, indices)

if __name__ == "__main__":
    main()
