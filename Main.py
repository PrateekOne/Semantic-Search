import nltk
from rank_bm25 import BM25Okapi
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, render_template, jsonify, url_for, session
import os
import requests
from PIL import Image
from googletrans import Translator
import time # For time-based limits

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Set a secret key for session management
app.secret_key = 'your_super_secret_key_here' # Replace with a strong, random key in production
# Example for production: app.secret_key = os.urandom(24)

# --- NLTK Setup ---
# Ensure you have these NLTK data downloaded. If not, run:
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Global Model and Data Variables (Initialized to None/Empty) ---
# These will be populated by load_all_models_and_data()
device = None
flan_tokenizer = None
flan_model = None
model = None # SentenceTransformer
hinglish_translator_tokenizer = None
hinglish_translator_model = None
google_translator = None

df = pd.DataFrame() # Initialize df as empty
dense_embeddings = np.array([]) # Initialize dense_embeddings as empty
bm25 = None # Initialize bm25 as None

# NEW: Constants for summarization
MAX_INPUT_LENGTH = 512 # Max tokens for input to Flan-T5
MAX_SUMMARY_LENGTH = 150 # Max tokens for generated summary (aiming for 2-10 sentences)

# Paths to your data files
csv_path = "Book_Details.csv"
json_path = "Book_Details_processed_embeddings.json"

# --- Model and Data Loading Function ---
def load_all_models_and_data():
    global device, flan_tokenizer, flan_model, model, \
           hinglish_translator_tokenizer, hinglish_translator_model, \
           google_translator, df, dense_embeddings, bm25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for models: {device}")

    # Flan-T5
    print("Loading Flan-T5 model...")
    flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    print("Flan-T5 model loaded.")

    # Sentence Transformer
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("SentenceTransformer model loaded.")

    # NEW: Hinglish-to-English Translator Model (rudrashah/RLM-hinglish-translator)
    print("Loading Hinglish-to-English translator model...")
    try:
        hinglish_translator_tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
        hinglish_translator_model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator").to(device)
        print("rudrashah/RLM-hinglish-translator model loaded for translation.")
    except Exception as e:
        print(f"Error loading rudrashah/RLM-hinglish-translator model: {e}. Attempting to use googletrans as a fallback.")
        try:
            google_translator = Translator()
            print("googletrans Translator initialized as fallback.")
        except Exception as ge:
            print(f"Error initializing googletrans: {ge}. No translation model available.")

    # --- Data Loading ---
    print("Loading data from CSV and JSON...")
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(f)
        df = df.iloc[:16226] # Limit to the first 16226 rows

        # --- Add new 'summary' column ---
        df['summary'] = None # Initialize with None for caching

        # --- Image Path Handling ---
        if 'cover_image_uri' not in df.columns:
            print(f"Warning: 'cover_image_uri' column not found in {csv_path}. Using 'placeholder.png' as default.")
            df['cover_image_uri'] = 'placeholder.png'
        df['cover_image_uri'] = df['cover_image_uri'].fillna('placeholder.png')

        # Check for 'author' column for better Open Library search if it exists
        if 'author' not in df.columns:
            print(f"Warning: 'author' column not found in {csv_path}. Open Library searches might be less precise.")
            df['author'] = '' # Add an empty author column if it doesn't exist

        corpus = df["book_details"].fillna("").astype(str).tolist()
        print(f"Loaded {len(df)} book entries from {csv_path}")

        tokenized_corpus = [
            [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words] for doc in corpus
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 model initialized.")

        with open(json_path, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
        dense_embeddings = np.array([entry["embedding"] for entry in embeddings_data])
        print(f"Loaded {len(dense_embeddings)} embeddings from {json_path}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}. Please ensure '{csv_path}' and '{json_path}' exist.")
    except Exception as e:
        print(f"Error loading data or initializing models: {e}")
    print("All models and data loaded successfully.")

# --- Utility Functions ---
def translate_query(query):
    # Try the new Hinglish-to-English model first
    if hinglish_translator_model and hinglish_translator_tokenizer:
        try:
            input_formatted = f"Hinglish:\n{query}\n\nEnglish:\n"
            inputs = hinglish_translator_tokenizer(input_formatted, return_tensors="pt").input_ids.to(device)

            output = hinglish_translator_model.generate(
                inputs,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True,
                pad_token_id=hinglish_translator_tokenizer.eos_token_id
            )
            translated_text_full = hinglish_translator_tokenizer.decode(output[0], skip_special_tokens=True)

            if "\n\nEnglish:\n" in translated_text_full:
                translated_text = translated_text_full.split("\n\nEnglish:\n", 1)[1].strip()
            else:
                translated_text = translated_text_full.strip()

            print(f"Translated '{query}' to '{translated_text}' using rudrashah/RLM-hinglish-translator.")
            return translated_text
        except Exception as e:
            print(f"rudrashah/RLM-hinglish-translator translation failed for '{query}': {e}. Falling back to googletrans.")

    # Fallback to googletrans if the new model fails or is not loaded
    if google_translator:
        try:
            translated = google_translator.translate(query, dest='en')
            if translated and translated.text:
                print(f"Translated '{query}' to '{translated.text}' using googletrans.")
                return translated.text
            else:
                print(f"googletrans returned empty translation for '{query}'. Returning original query.")
                return query
        except Exception as e:
            print(f"googletrans translation failed for '{query}': {e}. Returning original query.")
            return query
    else:
        print("No translation model loaded. Skipping translation and using original query.")
        return query

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

def smart_search(query, top_k=10, alpha=0.4, bm25_penalty=0.5):
    if bm25 is None or dense_embeddings.size == 0 or df.empty:
        print("Search components not initialized. Cannot perform search.")
        return "Error", np.array([])

    query_tokens = [word for word in word_tokenize(query.lower()) if word.isalnum() and word not in stop_words]

    bm25_scores = bm25.get_scores(query_tokens)
    bm25_scores *= bm25_penalty
    bm25_scores_norm = normalize_scores(bm25_scores)

    query_embedding = model.encode([query], batch_size=1)[0]
    dense_scores = cosine_similarity([query_embedding], dense_embeddings)[0]
    dense_scores_norm = normalize_scores(dense_scores)

    min_len = min(len(bm25_scores_norm), len(dense_scores_norm))
    # Ensure scores are truncated to the length of the shortest array if needed
    bm25_scores_norm = bm25_scores_norm[:min_len]
    dense_scores_norm = dense_scores_norm[:min_len]

    hybrid_scores = alpha * dense_scores_norm + (1 - alpha) * bm25_scores_norm

    if 'average_rating' in df.columns:
        ratings = df['average_rating'].fillna(df['average_rating'].mean())[:min_len]
        rating_boost = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-9)
        hybrid_scores += 0.05 * rating_boost

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return "Hybrid", top_indices

def truncate_text(text, max_length=150):
    words = text.split()
    if len(words) > max_length:
        return " ".join(words[:max_length]) + "..."
    return text

def format_prompt_for_flan(user_query, top_indices, df_data):
    # Ensure df_data is passed correctly
    truncated_query = truncate_text(user_query, max_length=50)
    prompt = f"User Query: {truncated_query}\n\nCandidate Book Titles:\n"
    for i, idx in enumerate(top_indices):
        # Use df_data instead of global df
        title = df_data.iloc[idx]["book_title"]
        prompt += f"{i+1}. {title}\n"

    prompt += (
        "\nInstruction: From the 'Candidate Book Titles' listed above, select the SINGLE book title "
        "that is MOST relevant to the 'User Query'.\n"
        "Your response MUST contain ONLY the exact book title from the list.\n"
        "DO NOT include any numbers, explanations, summaries, introductory phrases, or additional text.\n"
        "Return ONLY the book title.\n"
        "Absolutely no other words or characters should be in your response."
    )
    return prompt

def generate_flan_response(prompt):
    input_ids = flan_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # Ensure max_token_length is respected
    input_ids = input_ids[:, :MAX_INPUT_LENGTH]
    outputs = flan_model.generate(input_ids, max_new_tokens=300, temperature=0.7, num_return_sequences=1)
    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def generate_summary(text):
    global flan_tokenizer, flan_model, device # Ensure access to global models

    if pd.isna(text) or not text.strip():
        return "" # Return empty string for NaN or empty descriptions

    # A more specific prompt to guide for detail and length
    prompt = f"Summarize the following book description in 2 to 10 sentences, highlighting key plot points and themes:\n\n{text}\n\nSummary:"
    
    inputs = flan_tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids.to(device)
    
    # Generate summary
    summary_ids = flan_model.generate(
        inputs.input_ids,
        num_beams=4,
        min_length=30, # Minimum length of the summary in tokens
        max_new_tokens=MAX_SUMMARY_LENGTH, # Maximum number of new tokens to generate
        early_stopping=True,
        no_repeat_ngram_size=2 # Helps prevent repetitive phrases
    )
    summary = flan_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process to ensure it looks like 2-10 lines (soft constraint)
    # This is a heuristic, real line breaks depend on content and rendering
    sentences = nltk.sent_tokenize(summary)
    # If the generated summary is very short (e.g., less than 30 words) AND original is substantial, return original
    if len(sentences) < 2 and len(summary.split()) < 30 and len(text.split()) > 50:
        return text
    elif len(sentences) > 10:
        # Truncate to maximum 10 sentences if too long
        summary = " ".join(sentences[:10]) + "..."
    
    return summary.strip()


# --- Open Library API Integration ---
def get_openlibrary_url(title, author=None):
    base_url = "https://openlibrary.org"
    search_api_url = f"{base_url}/search.json"

    params = {'q': title}
    if author:
        params['author'] = author

    try:
        response = requests.get(search_api_url, params=params, timeout=5) # 5-second timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if data and 'docs' in data and len(data['docs']) > 0:
            first_doc = data['docs'][0]
            if 'key' in first_doc:
                return f"{base_url}{first_doc['key']}" # e.g., https://openlibrary.org/works/OL12345W
            elif 'isbn' in first_doc and len(first_doc['isbn']) > 0:
                return f"{base_url}/isbn/{first_doc['isbn'][0]}"
            elif 'olid' in first_doc and len(first_doc['olid']) > 0:
                return f"{base_url}/books/{first_doc['olid'][0]}"

        # Fallback to a generic search URL if no specific book key/ID is found
        return f"{base_url}/search?q={requests.utils.quote(title)}"

    except requests.exceptions.RequestException as e:
        print(f"Error querying Open Library API for '{title}': {e}")
        return f"{base_url}/search?q={requests.utils.quote(title)}"
    except Exception as e:
        print(f"Unexpected error processing Open Library response for '{title}': {e}")
        return f"{base_url}/search?q={requests.utils.quote(title)}"


# --- Flask Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    original_query = query
    translated_query = None
    processed_query = query # Initialize processed_query

    if query:
        processed_query = translate_query(query)
        # Check if translation actually changed the query
        if processed_query != query:
            translated_query = processed_query

        # Perform the initial hybrid search here to get top_indices
        method, top_indices = smart_search(processed_query, top_k=10)
        session['top_indices'] = top_indices.tolist() # Store as list in session
        session['processed_query'] = processed_query
        # Store the start time for summary generation
        session['summary_generation_start_time'] = time.time()
    else:
        session['top_indices'] = []
        session['processed_query'] = ""
        session['summary_generation_start_time'] = time.time() # Even if no query, set time

    return render_template(
        'search.html',
        query=original_query, # Keep original query for input field
        translated_query=translated_query,
        # results will be loaded via AJAX
        # flan_recommendation will be loaded via AJAX
    )

@app.route('/get_hybrid_results', methods=['GET'])
def get_hybrid_results():
    # Retrieve top_indices and processed_query from session
    top_indices = session.get('top_indices', [])
    if not top_indices:
        return jsonify([])

    results_data = []
    for idx in top_indices:
        book = df.iloc[idx]
        results_data.append({
            "index": int(idx), # Send index back to identify the element
            "title": book['book_title'],
            # For initial display, send a short snippet or "Generating summary..."
            # The full summary/original will be loaded by get_book_details
            "initial_snippet": truncate_text(book['book_details'], max_length=150)
        })
    return jsonify(results_data)

@app.route('/get_book_details/<int:book_index>', methods=['GET'])
def get_book_details(book_index):
    global df # Declare df as global to modify it
    
    if book_index not in df.index:
        return jsonify({"error": "Book not found"}), 404

    book = df.iloc[book_index]
    image_url = book['cover_image_uri']

    if not image_url.startswith('http') and not image_url.startswith('/'):
        image_url = url_for('static', filename=f'images/{image_url}')

    openlibrary_book_url = get_openlibrary_url(book['book_title'], book.get('author', None))

    book_summary = df.loc[book_index, 'summary'] # Get summary from df
    original_description = book['book_details'] # Always send the original for toggle

    # Check for summary and time limit
    summary_generation_start_time = session.get('summary_generation_start_time', time.time())
    
    # 30 seconds for the entire set of summaries for this search session
    if pd.isna(book_summary) and (time.time() - summary_generation_start_time) < 30:
        try:
            print(f"Generating summary for book index {book_index}...")
            generated_summary = generate_summary(original_description)
            df.loc[book_index, 'summary'] = generated_summary # Store in DataFrame
            book_summary = generated_summary
            print(f"Summary generated for book index {book_index}.")
        except Exception as e:
            print(f"Error generating summary for book index {book_index}: {e}")
            book_summary = original_description # Fallback to original
    elif pd.isna(book_summary) and (time.time() - summary_generation_start_time) >= 30:
        print(f"Time limit for summary generation exceeded. Returning original description for book index {book_index}.")
        book_summary = original_description # Fallback if time limit passed

    # If summary was still None/NaN (e.g., description too short or generation failed early), use original
    if pd.isna(book_summary) or not book_summary.strip():
        book_summary = original_description

    return jsonify({
        "index": book_index,
        "image": image_url,
        "url": openlibrary_book_url,
        "summary": book_summary,            # This will be the generated summary or original
        "original_description": original_description # This is always the full original
    })

@app.route('/get_flan_recommendation', methods=['GET'])
def get_flan_recommendation():
    top_indices = session.get('top_indices', [])
    processed_query = session.get('processed_query', '')

    if not top_indices or not processed_query:
        return jsonify({"recommendation": "No search performed or results available for recommendation."})

    flan_recommendation = "No specific recommendation generated."

    flan_prompt = format_prompt_for_flan(processed_query, top_indices, df)
    try:
        # Simulate a delay for demonstration (remove in production if not needed)
        # time.sleep(2) 
        flan_output = generate_flan_response(flan_prompt)
        flan_recommendation = flan_output
    except Exception as e:
        print(f"Error generating FLAN-T5 recommendation: {e}")
        # Fallback to the first result if Flan-T5 fails
        if top_indices:
            first_book_title = df.iloc[top_indices[0]]['book_title']
            flan_recommendation = f"Error generating recommendation. Top hybrid result: {first_book_title}"
        else:
            flan_recommendation = "Error generating recommendation."

    return jsonify({"recommendation": flan_recommendation})

if __name__ == '__main__':
    # --- Data File and Static Directory Checks ---
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please create it or update the path.")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please create it or update the path.")

    # Ensure static/images directory exists and create a placeholder image
    static_images_path = os.path.join(app.root_path, 'static', 'images')
    os.makedirs(static_images_path, exist_ok=True)
    placeholder_path = os.path.join(static_images_path, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        print(f"Warning: 'placeholder.png' not found in {static_images_path}. Creating a dummy file.")
        try:
            # Create a simple transparent PNG placeholder (e.g., 80x120 pixels)
            img = Image.new('RGBA', (80, 120), (255, 255, 255, 0))
            img.save(placeholder_path, 'PNG')
            print("Created 'placeholder.png' for missing book covers.")
        except ImportError:
            print("Pillow (PIL) not installed. Please install it (`pip install Pillow`) to create a dummy placeholder image, or create 'placeholder.png' manually in your static/images folder.")
        except Exception as e:
            print(f"Error creating placeholder.png: {e}. Please create it manually.")

    # Load all models and data ONLY ONCE when the script starts
    load_all_models_and_data()

    # Run the Flask app
    # Set use_reloader=False to prevent models from loading multiple times in debug mode
    app.run(debug=True, use_reloader=False)