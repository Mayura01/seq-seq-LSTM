import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

def tokenize_texts(texts):
    word_to_index = {}
    index = 1
    for text in texts:
        for word in text.split():
            if word.lower() not in word_to_index:
                word_to_index[word.lower()] = index
                index += 1
    return word_to_index

def generate_sequences(texts, word_to_index):
    sequences = []
    for text in texts:
        sequence = []
        for word in text.split():
            sequence.append(word_to_index.get(word.lower(), 0))
        sequences.append(sequence)
    return sequences

def custom_pad_sequences(sequences, max_seq_length):
    padded_sequences = np.zeros((len(sequences), max_seq_length), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        if len(sequence) > max_seq_length:
            padded_sequences[i, :] = sequence[:max_seq_length]
        else:
            padded_sequences[i, :len(sequence)] = sequence
    return padded_sequences

# Define global variables
max_seq_length = 100

# Preprocess data
texts = ["How are you?"]
word_to_index = tokenize_texts(texts)
sequences = generate_sequences(texts, word_to_index)
padded_sequences = custom_pad_sequences(sequences, max_seq_length)
print("Preprocess complete...")

# input-output pairs for seq2seq model
input_data = keras_pad_sequences(padded_sequences[:, :-1], maxlen=max_seq_length - 1, padding='post')
target_data = keras_pad_sequences(padded_sequences[:, 1:], maxlen=max_seq_length - 1, padding='post')
vocab_size = len(word_to_index) + 1
print("Done with input-output pairs for seq2seq model...")

# Define seq2seq model
embedding_dim = 128
units = 256

# Define encoder and decoder inputs
encoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))
decoder_inputs = tf.keras.Input(shape=(max_seq_length - 1,))

# Define encoder and decoder layers
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_embedding)
encoder_states = [state_h_enc, state_c_enc]

decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define model
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Load weights
try:
    model.load('M68.h5')
    print("Loaded model weights successfully...")
except:
    print("No existing model weights found...")

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

predictions = model.predict([input_data, target_data])

# Define index_to_word dictionary
index_to_word = {index: word for word, index in word_to_index.items()}

def decode_sequence(sequence, index_to_word):
    decoded_words = []
    for token in sequence:
        if np.isscalar(token):
            if token != 0:
                word = index_to_word.get(token, '')
                decoded_words.append(word)
        else:
            for idx in token:
                if idx != 0:
                    word = index_to_word.get(idx, '')
                    decoded_words.append(word)
    return ' '.join(decoded_words)



decoded_predictions = [decode_sequence(seq, index_to_word) for seq in predictions]

for i, input_text in enumerate(texts):
    print("Input Text:", input_text)
    print("Generated Text:", decoded_predictions[i])
