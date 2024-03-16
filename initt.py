import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import pickle

# Define the functions for the model
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, inputs):
        return self.W[inputs]

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        self.h = None
        self.c = None

    def forward(self, inputs, states):
        self.h, self.c = states
        z = np.column_stack((inputs, self.h))
        f = sigmoid(np.dot(z, self.Wf) + self.bf)
        i = sigmoid(np.dot(z, self.Wi) + self.bi)
        c_hat = np.tanh(np.dot(z, self.Wc) + self.bc)
        o = sigmoid(np.dot(z, self.Wo) + self.bo)
        self.c = f * self.c + i * c_hat
        self.h = o * np.tanh(self.c)
        return self.h, (self.h, self.c)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_scores = np.exp(x - np.max(x))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, inputs):
        return softmax(np.dot(inputs, self.W) + self.b)

class Seq2Seq:
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        self.encoder_embedding = Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = LSTM(embedding_dim, hidden_size)
        self.decoder_embedding = Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = LSTM(embedding_dim, hidden_size)
        self.decoder_dense = Dense(hidden_size, vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding.forward(encoder_inputs)
        encoder_states = self.encoder_lstm.forward(encoder_embedded, (np.zeros((encoder_inputs.shape[0], self.encoder_lstm.hidden_size)), np.zeros((encoder_inputs.shape[0], self.encoder_lstm.hidden_size))))
        # Assuming decoder_inputs is of the same shape as encoder_inputs
        decoder_embedded = self.decoder_embedding.forward(decoder_inputs)
        decoder_outputs, _ = self.decoder_lstm.forward(decoder_embedded, encoder_states)
        output = self.decoder_dense.forward(decoder_outputs)
        return output


# Define MongoDB connection
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected and got the dataset...")

# Tokenize texts
def tokenize_texts(texts):
    word_to_index = {}
    index = 1
    for text in texts:
        for word in text.split():
            if word.lower() not in word_to_index:
                word_to_index[word.lower()] = index
                index += 1
    return word_to_index

# Generate sequences
def generate_sequences(texts, word_to_index):
    sequences = []
    for text in texts:
        sequence = []
        for word in text.split():
            sequence.append(word_to_index.get(word.lower(), 0))
        sequences.append(sequence)
    return sequences

# Padding sequences
def pad_sequences(sequences, max_seq_length):
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
padded_sequences = pad_sequences(sequences, max_seq_length)
print("Preprocessing complete...")

# Input-output pairs for Seq2Seq model
input_data = padded_sequences[:, :-1]
target_data = padded_sequences[:, 1:]
vocab_size = len(word_to_index) + 1
print("Done with input-output pairs for Seq2Seq model...")

# Initialize the Seq2Seq model
embedding_dim = 128
hidden_size = 256
seq2seq_model = Seq2Seq(vocab_size, embedding_dim, hidden_size)

# Load existing model weights if available
try:
    with open('model_weights.pkl', 'rb') as f:
        seq2seq_model = pickle.load(f)
    print("Loaded model weights successfully...")
except Exception as e:
    print("No existing model weights found:", e)

# Train the model
# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Define loss function
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

# Define number of epochs
epochs = 10

# Define batch size
batch_size = 32

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Shuffle input data for each epoch
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    shuffled_input_data = input_data[indices]
    shuffled_target_data = target_data[indices]
    
    # Iterate over batches
    for i in range(0, len(input_data), batch_size):
        batch_input = shuffled_input_data[i:i+batch_size]
        batch_target = shuffled_target_data[i:i+batch_size]
        
        # Forward pass
        # Forward pass
        with tf.GradientTape() as tape:
            predictions = seq2seq_model.forward(batch_input, batch_input)  # assuming batch_input serves as both encoder and decoder inputs
            loss = loss_function(batch_target, predictions)
        
        # Backpropagation
        gradients = tape.gradient(loss, seq2seq_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, seq2seq_model.trainable_variables))
        
        # Print loss for monitoring
        print(f"Batch {i//batch_size + 1}/{len(input_data)//batch_size}, Loss: {loss.numpy()}")

# Save the model weights
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(seq2seq_model, f)
print("Model weights saved successfully...")
