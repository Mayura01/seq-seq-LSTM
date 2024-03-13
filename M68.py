import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected and got the data set...")

# Preprocess data
texts = [conversation['body'] for conversation in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_seq_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
print("Preprocess complete...")

# input-output pairs for seq2seq model
input_data = padded_sequences[:, :-1]  # Input sequence
target_data = padded_sequences[:, 1:]  # Target sequence shifted by one timestep
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
print("Done with input-output pairs for seq2seq model...")

# Define seq2seq model
embedding_dim = 128
units = 256

encoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([input_data, input_data], target_data, batch_size=8, epochs=10, validation_split=0.2)


# text to test the model
input_text = "How are you?"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_seq_length - 1, padding='post')

# Encode the input sequence
encoder_input = input_sequence
encoder_output, state_h_enc, state_c_enc = encoder_lstm(encoder_embedding(encoder_input))
encoder_states = [state_h_enc, state_c_enc]

# Initialize the decoder input with a start token
decoder_input = np.zeros((1, max_seq_length - 1))
decoder_input[0, 0] = tokenizer.word_index['<start>']

# output sequence
output_sequence = []
for _ in range(max_seq_length - 1):
    decoder_output, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding(decoder_input), initial_state=encoder_states)
    decoder_output = decoder_dense(decoder_output)
    decoder_input[0, 1:] = np.argmax(decoder_output, axis=-1)
    encoder_states = [state_h_dec, state_c_dec]
    output_sequence.append(np.argmax(decoder_output, axis=-1)[0, 0])

# Decode the output sequence into text
output_text = tokenizer.sequences_to_texts([output_sequence])

print("Input Text:", input_text)
print("Generated Text:", output_text)
