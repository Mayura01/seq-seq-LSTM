import numpy as np
import tensorflow as tf
from pymongo import MongoClient

# Load data
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected and got the dataset...")

def tokenize_texts(texts):
    word_to_index = {}
    index = 1
    for text in texts:
        for word in text.split():
            if word.lower() not in word_to_index:
                word_to_index[word.lower()] = index
                index += 1
    return word_to_index

# Sequence Generation
def generate_sequences(texts, word_to_index):
    sequences = []
    for text in texts:
        sequence = []
        for word in text.split():
            sequence.append(word_to_index.get(word.lower(), 0))
        sequences.append(sequence)
    return sequences

# Padding
def custom_pad_sequences(sequences, max_seq_length):
    padded_sequences = np.zeros((len(sequences), max_seq_length), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        if len(sequence) > max_seq_length:
            padded_sequences[i, :] = sequence[:max_seq_length]
        else:
            padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences

# Preprocess data
texts = [conversation['body'] for conversation in data]
word_to_index = tokenize_texts(texts)
sequences = generate_sequences(texts, word_to_index)
max_seq_length = 100
padded_sequences = custom_pad_sequences(sequences, max_seq_length)
print("Preprocess complete...")

# Input-output pairs for seq2seq model
input_data = padded_sequences[:, :-1]
target_data = padded_sequences[:, 1:]
vocab_size = len(word_to_index) + 1
print("Done with input-output pairs for seq2seq model...")

# Define seq2seq model
embedding_dim = 128
units = 256

# Encoder and decoder inputs
encoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))
decoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))

# Encoder and decoder layers
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_embedding)
encoder_states = [state_h_enc, state_c_enc]

decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

try:
    model = tf.keras.models.load_model('M68.h5')
    print("Loaded model successfully...")
except OSError:
    print("No existing model found...")


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([input_data, input_data], target_data, batch_size=16, epochs=1, validation_split=0.2)

model.save('M68.h5')
