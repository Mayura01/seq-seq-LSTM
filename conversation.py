import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

# Define the custom layer
class NotEqual(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.not_equal(inputs[0], inputs[1])

# Register the custom object
tf.keras.utils.get_custom_objects()['NotEqual'] = NotEqual

# Debugging prints
print("Custom layer registered.")

# Load the model
model = tf.keras.models.load_model('M68.h5')

# Debugging prints
print("Model loaded successfully.")

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

example_conversations = [
    "Hi there!",
    "How are you?",
    "What are you up to?",
    "I'm just coding. How about you?",
    "I'm reading a book.",
    "That sounds interesting."
]

word_to_index = tokenize_texts(example_conversations)
index_to_word = {index: word for word, index in word_to_index.items()}

# encode input text
def encode_input_text(input_text, word_to_index, max_seq_length):
    input_sequence = [word_to_index.get(word.lower(), 0) for word in input_text.split()]
    padded_input_sequence = keras_pad_sequences([input_sequence], maxlen=max_seq_length - 1, padding='post')
    return padded_input_sequence

# decode output sequence
def decode_output_sequence(output_sequence, index_to_word):
    decoded_words = [index_to_word[idx] for idx in output_sequence if idx != 0]
    decoded_text = ' '.join(decoded_words)
    return decoded_text


max_seq_length = 100
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    encoded_input = encode_input_text(user_input, word_to_index, max_seq_length)
    predicted_output = model.predict([encoded_input, encoded_input])
    output_sequence = np.argmax(predicted_output[0], axis=-1)
    decoded_response = decode_output_sequence(output_sequence, index_to_word)
    
    print("Bot:", decoded_response)
