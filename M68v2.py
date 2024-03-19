import tensorflow as tf
from pymongo import MongoClient

# Load data
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['Cornell_Movie_Dialog_Corpus']
collection = db['Dialog_Corpus_1']
data = collection.find()
print("Connected and got the dataset...")

# Preprocess data
questions = []
answers = []

for conversation in data:
    questions.append(conversation['question'])
    answers.append(conversation['answer'])

# Tokenize the questions and answers
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Define maximum sequence length
max_seq_length = 100  # Set your desired maximum sequence length here

# Define a generator function
def data_generator(questions, answers, tokenizer, max_seq_length, batch_size):
    num_samples = len(questions)
    while True:
        for start in range(0, num_samples, batch_size):
            encoder_input_data = []
            decoder_input_data = []
            decoder_target_data = []
            end = min(start + batch_size, num_samples)
            for i in range(start, end):
                # Tokenize and pad encoder input sequence
                encoder_seq = tokenizer.texts_to_sequences([questions[i]])[0]
                encoder_seq = tf.keras.preprocessing.sequence.pad_sequences([encoder_seq], maxlen=max_seq_length, padding='post')
                encoder_input_data.append(encoder_seq)

                # Tokenize and pad decoder input sequence
                decoder_seq = tokenizer.texts_to_sequences([answers[i]])[0]
                decoder_seq = tf.keras.preprocessing.sequence.pad_sequences([decoder_seq], maxlen=max_seq_length, padding='post')
                decoder_input_data.append(decoder_seq)

                # Shift decoder target sequence for training
                decoder_target_seq = decoder_seq[:, 1:]
                decoder_target_data.append(decoder_target_seq)
            
            # Convert lists to TensorFlow tensors
            encoder_input_data = tf.constant(encoder_input_data)
            decoder_input_data = tf.constant(decoder_input_data)
            decoder_target_data = tf.constant(decoder_target_data)

            yield ([encoder_input_data, decoder_input_data], decoder_target_data)


# Define hyperparameters
batch_size = 16
epochs = 10

# Define your model architecture
def M68v2(encoder_inputs, decoder_inputs, vocab_size):
    embedding_dim = 128
    units = 256

    # Encoder layers
    encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    _, state_h_enc, state_c_enc = tf.keras.layers.LSTM(units, return_state=True)(encoder_embedding)
    encoder_states = [state_h_enc, state_c_enc]

    # Decoder layers
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Create and compile the model
encoder_inputs = tf.keras.Input(shape=(max_seq_length,))
decoder_inputs = tf.keras.Input(shape=(max_seq_length,))
model = M68v2(encoder_inputs, decoder_inputs, vocab_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generator
model.fit(data_generator(questions, answers, tokenizer, max_seq_length, batch_size), steps_per_epoch=len(questions)//batch_size, epochs=epochs)

# Save the model
model.save('M68v2.keras')
print("Model saved after training.")
