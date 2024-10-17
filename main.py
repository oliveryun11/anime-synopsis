import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the data
df = pd.read_csv('data/anime-dataset-2023.csv')
data = df[['Genres', 'Synopsis']]

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

# Set maximum lengths for genres and synopses
MAX_GENRE_LENGTH = 32 # 23
MAX_SYNOPSIS_LENGTH = 1024 # 920

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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
batch_size = 4

# Training loop
num_epochs = 3 
print("training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[batch:batch+batch_size].to(device)
        batch_masks = train_masks[batch:batch+batch_size].to(device)
        batch_outputs = train_outputs[batch:batch+batch_size].to(device)

        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_outputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / (len(train_inputs) / batch_size)
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")

