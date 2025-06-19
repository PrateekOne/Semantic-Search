# 📚 Semantic Book Search

A powerful hybrid book search engine that combines semantic understanding with keyword relevance. Built using BM25, MiniLM embeddings, and FLAN-T5 for natural language recommendations — all served through a clean, fast Flask web app.

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/PrateekOne/Semantic-Search.git
cd Semantic-Search
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Make sure you also install the following NLTK data if you haven't already:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 🚀 Usage

### Step 1: Generate Embeddings

Run the following script to preprocess book data and generate the required embeddings:

```bash
python embed.py
```

This will create a file named `Book_Details_processed_embeddings.json` containing dense sentence embeddings and metadata.

### Step 2: Launch the Flask App

Run the main web interface:

```bash
python main.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

## 🔍 Pipeline Overview

1. **Preprocessing (`embed.py`)**:

   * Loads book metadata from `Book_Details.csv`
   * Tokenizes text and creates BM25 index
   * Generates semantic embeddings using `all-MiniLM-L6-v2`
   * Saves results in `Book_Details_processed_embeddings.json`

2. **Search Flow (`main.py`)**:

   * Accepts a user query in any language (supports Romanized Hindi)
   * Detects and translates to English if necessary
   * Retrieves top documents using a hybrid BM25 + semantic score
   * FLAN-T5 generates a personalized book recommendation
   * Displays results in a Tailwind-styled web UI

---

## ✨ Features

* 🔍 **Hybrid Search**: Combines sparse (BM25) and dense (MiniLM) retrieval
* 🌐 **Multilingual Query Handling**: Detects and translates non-English queries
* 🤖 **LLM Recommendation**: Uses FLAN-T5 to generate query-aware book summaries
* 📚 **OpenLibrary Integration**: Direct links to book pages
* 💡 **Fast Local Deployment**: No external API needed after setup

---

## 📁 Project Structure

| File/Folder                              | Description                         |
| ---------------------------------------- | ----------------------------------- |
| `main.py`                                | Flask backend for search interface  |
| `embed.py`                               | Script to preprocess and embed data |
| `Book_Details.csv`                       | Raw book dataset                    |
| `Book_Details_processed_embeddings.json` | Generated metadata + embeddings     |
| `templates/index.html`                   | Web UI (HTML)                       |
| `static/style.css`                       | Tailwind CSS styling                |
| `requirements.txt`                       | Python dependencies                 |

---

## 🧠 Technologies Used

* Python, Flask
* SentenceTransformers (MiniLM)
* BM25 (Rank-BM25)
* FLAN-T5-Large (Hugging Face Transformers)
* NLTK, NumPy, Pandas
* Googletrans / NLLB for translation
* Tailwind CSS

---
