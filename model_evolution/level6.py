import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(chunk_number, chunk_size):
    client = MongoClient('mongodb://127.0.0.1:27017/')
    db = client['reddit_dataset']
    collection = db['comments_chunk_' + str(chunk_number)]

    data = collection.find().limit(chunk_size)
    
    input_data = []
    target_data = []
    for conversation in data:
        input_data.append(conversation['body'])
        target_data.append(conversation['body'])
    
    return input_data, target_data


def build_model(vocab_size, embedding_dim, units):
    # Define encoder
    encoder_inputs = tf.keras.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Define decoder
    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(model, input_data, target_data):
    encoder_input_data = input_data[:, :-1]
    decoder_input_data = input_data[:, 1:]
    decoder_target_data = target_data[:, 1:]

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=16, epochs=1, validation_split=0.2)


def preprocess_data(texts, max_seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_to_index = tokenizer.word_index 
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    
    return padded_sequences, word_to_index


def load_and_preprocess_data(chunk_number, chunk_size, max_seq_length):
    input_data, target_data = load_data(chunk_number, chunk_size)
    input_sequences, word_to_index = preprocess_data(input_data, max_seq_length)
    target_sequences, _ = preprocess_data(target_data, max_seq_length)
    
    return input_sequences, target_sequences, word_to_index


def main():
    vocab_size = 10000
    embedding_dim = 128
    units = 256
    total_chunks = 10
    chunk_size = 25000
    max_seq_length = 100

    input_data, target_data, word_to_index = load_and_preprocess_data(1, chunk_size, max_seq_length)
    vocab_size = len(word_to_index) + 1
    
    model = build_model(vocab_size, embedding_dim, units)

    # try:
    #     model = tf.keras.models.load_model('M68.keras')
    #     print("Loaded model successfully...")
    # except OSError:
    #     print("No existing model found. Training a new model...")

    train_model(model, input_data, target_data)

    model.save('M68.keras')
    print("Model saved after training chunk", 1)

if __name__ == "__main__":
    main()
