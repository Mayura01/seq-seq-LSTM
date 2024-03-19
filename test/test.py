import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pymongo import MongoClient

# Load data
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected and got the dataset...")

# Tokenize texts using GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token

# Define batch size
batch_size = 8

# Preprocess data and split into batches
texts = [conversation['body'] for conversation in data]
input_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

# Define model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))  # Resize model's token embeddings to match tokenizer

# Training parameters
num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    total_loss = 0
    for batch_texts in input_batches:
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        labels = inputs['input_ids'].clone()  # Use the same input for labels in language modeling

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(input_batches)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_chatbot_model')
print("Model saved after training chunk")
