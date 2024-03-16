import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the trained model
model = tf.keras.models.load_model('M68.h5')

def load_data(chunk_number, chunk_size):
    client = MongoClient('mongodb://127.0.0.1:27017/')
    db = client['reddit_dataset']
    collection = db['comments_chunk_' + str(chunk_number)]
    data = collection.find().limit(chunk_size)
    
    input_data = []
    for conversation in data:
        input_data.append(conversation['body'])
    
    return input_data

def preprocess_data(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_to_index = tokenizer.word_index 
    return word_to_index

def load_and_preprocess_data(chunk_number, chunk_size, max_seq_length):
    input_data = load_data(chunk_number, chunk_size)
    word_to_index = preprocess_data(input_data, max_seq_length)
    return word_to_index

# Function to preprocess input text
def preprocess_input_text(input_text, word_to_index, max_seq_length):
    # Tokenize input text
    input_sequence = [word_to_index.get(word, 0) for word in input_text.split()]
    # Pad sequences to fixed length
    input_sequence = pad_sequences([input_sequence], maxlen=max_seq_length, padding='post')
    return input_sequence

# Function to decode output tokens
def decode_output_tokens(output_tokens, index_to_word):
    decoded_output = ' '.join([index_to_word[token] for token in output_tokens if token != 0])
    return decoded_output

max_seq_length = 100
word_to_index = load_and_preprocess_data(1, 25000, max_seq_length)

# Reverse the word_to_index dictionary to create index_to_word
index_to_word = {index: word for word, index in word_to_index.items()}

# Preprocess input text (assuming input_text is a string)
input_text = "Hello"
input_sequence = preprocess_input_text(input_text, word_to_index, max_seq_length)

# Initialize variables for decoding loop
max_output_length = 50  # Maximum length of the output sequence
decoded_tokens = []

# Perform inference
output_sequence = tf.fill((1, 1), word_to_index['<start>']) # Start token for decoder

for _ in range(max_output_length):
    # Perform inference for one timestep
    output_logits, h, c = model([input_sequence, output_sequence])
    predicted_token = tf.argmax(output_logits[0, -1]).numpy()
    if predicted_token == word_to_index['<end>']:
        break
    decoded_tokens.append(predicted_token)
    output_sequence = tf.concat([output_sequence, [[predicted_token]]], axis=-1)

# Post-process output tokens to get response
response = decode_output_tokens(decoded_tokens, index_to_word)
print("Model response:", response)
