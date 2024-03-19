import numpy as np
import tensorflow as tf
from pymongo import MongoClient

model = tf.keras.models.load_model('M68.keras')
print("Model loaded successfully...")

client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected to the database...")

def tokenize_texts(texts):
    word_to_index = {}
    index = 1
    for text in texts:
        for word in text.split():
            if word.lower() not in word_to_index:
                word_to_index[word.lower()] = index
                index += 1
    word_to_index['<start>'] = index
    word_to_index['<end>'] = index + 1
    return word_to_index


texts = [conversation['body'] for conversation in data]
max_seq_length = 100
word_to_index = tokenize_texts(texts)
index_to_word = {index: word for word, index in word_to_index.items()}


def generate_text(input_text, word_to_index, index_to_word):
    input_sequence = [word_to_index.get(word.lower(), 0) for word in input_text.split()]
    input_sequence = input_sequence[:max_seq_length - 1]
    input_sequence += [0] * (max_seq_length - 1 - len(input_sequence))
    decoder_input = np.zeros((1, max_seq_length - 1), dtype=np.int32)
    decoder_input[0, 0] = word_to_index['<start>']
    
    for i in range(1, max_seq_length - 1):
        predictions = model.predict([np.array([input_sequence]), np.array(decoder_input)]).argmax(axis=-1)
        decoder_input[0, i] = predictions[0, i - 1]

        if predictions[0, i - 1] == word_to_index['<end>'] or i == max_seq_length - 2:
            break
    
    # Include the predicted tokens in the generated text
    generated_text_indices = [index for index in decoder_input[0] if index != 0]  # Exclude padding
    generated_text = ' '.join([index_to_word[index] for index in generated_text_indices if index != word_to_index['<start>']])  # Exclude <start>
    return generated_text


input_text = ''

while input_text != 'exit':
    input_text = input("you: ")
    generated_text = generate_text(input_text, word_to_index, index_to_word)
    print("Mayur: ",input_text)
    print("IntelliAi: ",generated_text)

