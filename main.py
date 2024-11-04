import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from config import *
from tqdm import tqdm

# Load the data
df = pd.read_csv('data/anime-dataset-2023.csv')
data = df[['Genres', 'Synopsis']]
data = data.drop(data[data['Synopsis'] == 'No description available for this anime.'].index)

# Initialize the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Tokenize and encode genres and synopses
def tokenize_and_encode(text, max_length):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

# Prepare input (genres) and output (synopses) data
input_ids = []
attention_masks = []
output_ids = []

for _, row in data.iterrows():
    genre_encoding = tokenize_and_encode(row['Genres'], MAX_GENRE_LENGTH)
    synopsis_encoding = tokenize_and_encode(row['Synopsis'], MAX_SYNOPSIS_LENGTH)
    
    input_ids.append(genre_encoding['input_ids'].squeeze())
    attention_masks.append(genre_encoding['attention_mask'].squeeze())
    output_ids.append(synopsis_encoding['input_ids'].squeeze())

# Convert to tensors
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
output_ids = torch.stack(output_ids)

# Split the data into train and test sets
train_inputs, test_inputs, train_masks, test_masks, train_outputs, test_outputs = train_test_split(
    input_ids, attention_masks, output_ids, test_size=0.2, random_state=42
)

# Load the BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
batch_size = BATCH_SIZE

# Training loop
num_epochs = NUM_EPOCHS 
print("Training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in tqdm(range(0, len(train_inputs), batch_size)):
        batch_inputs = train_inputs[batch:batch+batch_size].to(device)
        batch_masks = train_masks[batch:batch+batch_size].to(device)
        batch_outputs = train_outputs[batch:batch+batch_size].to(device)

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_outputs)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / (len(train_inputs) / batch_size)

    total_test_loss = 0
    model.eval()
    for batch in tqdm(range(0, len(test_inputs), batch_size)):
        batch_inputs = test_inputs[batch:batch+batch_size].to(device)
        batch_masks = test_masks[batch:batch+batch_size].to(device)
        batch_outputs = test_outputs[batch:batch+batch_size].to(device)

        with torch.no_grad():
            outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_outputs)
            loss = outputs.loss
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / (len(test_inputs) / batch_size)

    print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_train_loss:.4f} (Train), {avg_test_loss:.4f} (Test)")
    
    # Save the model after each epoch
    model_save_path = f'anime_synopsis_generator_epoch_{epoch+1}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Save the final model
final_model_save_path = 'model/anime_synopsis_generator_final.pt'
torch.save(model.state_dict(), final_model_save_path)
print(f"Final model saved to {final_model_save_path}")

# Save the tokenizer
tokenizer_save_path = 'model/anime_synopsis_tokenizer'
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")
