import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm # For progress bar

# Configuration
CSV_PATH = "Book_Details.csv"
OUTPUT_CSV_PATH = "Book_Details_with_summaries.csv" # New output file
MAX_INPUT_LENGTH = 512 # Max tokens for the input description
MAX_SUMMARY_LENGTH = 100 # Max tokens for the generated summary

# Load Flan-T5 for summarization
model_name = "google/flan-t5-large" # Using base model for faster summarization
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device for summarization: {device}")

def generate_summary(text):
    if pd.isna(text) or not text.strip():
        return "" # Return empty string for NaN or empty descriptions

    # Flan-T5 is good for various tasks, including summarization
    # For summarization, a common prompt format is 'summarize: <text>'
    prompt = f"summarize: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)

    # Generate summary
    summary_ids = model.generate(
        inputs.input_ids,
        num_beams=4,
        min_length=30, # Minimum length of the summary
        max_length=MAX_SUMMARY_LENGTH, # Maximum length of the summary
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    try:
        with open(CSV_PATH, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(f)
        df = df.iloc[:16226]
        print(f"Loaded {len(df)} entries from {CSV_PATH}")

        # Limit to the same number of rows as your Flask app for consistency
        df = df.iloc[:16226].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Add a new column for summaries
        df['book_summary'] = ""

        # Iterate and generate summaries
        # Using tqdm for a nice progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
            description = row['book_details'] # Assuming 'book_details' is the column with full descriptions
            if description: # Only summarize if description exists
                summary = generate_summary(description)
                df.at[index, 'book_summary'] = summary
            else:
                df.at[index, 'book_summary'] = "" # Ensure empty string if no description

        # Save the updated DataFrame
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nSummaries generated and saved to {OUTPUT_CSV_PATH}")

    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found. Please ensure the CSV file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()