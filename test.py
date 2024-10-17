import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the data
df = pd.read_csv('data/anime-dataset-2023.csv')
data = df[['Genres', 'Synopsis']]

# Initialize the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Tokenize without padding or truncation
def tokenize_without_padding(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors='pt'
    )

# Initialize variables to track max lengths
max_genre_length = 0
max_synopsis_length = 0

# First pass: determine max lengths
for _, row in data.iterrows():
    genre_encoding = tokenize_without_padding(row['Genres'])
    synopsis_encoding = tokenize_without_padding(row['Synopsis'])
    
    genre_length = genre_encoding['input_ids'].size(1)
    synopsis_length = synopsis_encoding['input_ids'].size(1)
    
    max_genre_length = max(max_genre_length, genre_length)
    max_synopsis_length = max(max_synopsis_length, synopsis_length)

print(f"Maximum genre length: {max_genre_length}")
print(f"Maximum synopsis length: {max_synopsis_length}")