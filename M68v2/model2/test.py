import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sec_model import Seq2Seq

# Sample data
data = [
    {"tag": "greeting", "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
     "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"]}
]

# Extract patterns, tags, and responses from data
patterns, responses = [], []
for entry in data:
    patterns.extend(entry["patterns"])
    responses.extend(entry["responses"])

# Text preprocessing
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns + responses)
vocab_size = len(tokenizer.word_index) + 1

# Padding sequences to a fixed length
max_seq_length = max(max(len(tokenizer.texts_to_sequences([pattern])[0]) for pattern in patterns),
                     max(len(tokenizer.texts_to_sequences([response])[0]) for response in responses))
encoder_inputs = pad_sequences(tokenizer.texts_to_sequences(patterns), maxlen=max_seq_length, padding='post')
decoder_inputs = pad_sequences(tokenizer.texts_to_sequences(responses), maxlen=max_seq_length, padding='post')
decoder_outputs = np.array([seq[1:] + [0] * (max_seq_length - len(seq) + 1) for seq in decoder_inputs])


embedding_dim = 128
hidden_units = 256

# Initialize the Seq2Seq model
seq2seq_model = Seq2Seq(embedding_dim, hidden_units)
model = seq2seq_model.build_model(max_seq_length, vocab_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split data into training and validation sets
split_index = int(len(encoder_inputs) * 0.8)
train_encoder_inputs = encoder_inputs[:split_index]
train_decoder_inputs = decoder_inputs[:split_index]
train_decoder_outputs = decoder_outputs[:split_index]
val_encoder_inputs = encoder_inputs[split_index:]
val_decoder_inputs = decoder_inputs[split_index:]
val_decoder_outputs = decoder_outputs[split_index:]


# Train the model
model.fit([train_encoder_inputs, train_decoder_inputs], train_decoder_outputs,
          validation_data=([val_encoder_inputs, val_decoder_inputs], val_decoder_outputs),
          batch_size=64, epochs=50)

# Save the trained model
model.save("seq2seq_model.h5")
