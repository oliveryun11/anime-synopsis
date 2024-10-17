import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from config import *

def load_model_and_tokenizer():
    # Load the trained model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.load_state_dict(torch.load('model/anime_synopsis_generator_final.pt', map_location=torch.device('cpu')))
    model.eval()

    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained('model/anime_synopsis_tokenizer')

    return model, tokenizer

def generate_synopsis(genres, model, tokenizer, max_length=150):
    # Tokenize the input genres
    inputs = tokenizer.encode_plus(
        genres,
        add_special_tokens=True,
        max_length=MAX_GENRE_LENGTH, 
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Generate the synopsis
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated synopsis
    synopsis = tokenizer.decode(output[0], skip_special_tokens=True)
    return synopsis

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Example usage
    genres = "Action, Adventure, Fantasy"
    generated_synopsis = generate_synopsis(genres, model, tokenizer)
    print(f"Genres: {genres}")
    print(f"Generated Synopsis: {generated_synopsis}")

