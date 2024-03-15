import tensorflow as tf
from pymongo import MongoClient

# Define the functions for the model
class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.encoder_lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)
        self.decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.decoder_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, encoder_inputs, decoder_inputs):
        encoder_embedded = self.encoder_embedding(encoder_inputs)
        encoder_outputs, state_h_enc, state_c_enc = self.encoder_lstm(encoder_embedded)
        encoder_states = [state_h_enc, state_c_enc]

        decoder_embedded = self.decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = self.decoder_lstm(decoder_embedded, initial_state=encoder_states)
        output = self.decoder_dense(decoder_outputs)
        return output

# Define MongoDB connection
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['reddit_dataset']
collection = db['comments_chunk_1']
data = collection.find()
print("Connected and got the dataset...")

# Tokenize texts
def tokenize_texts(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return sequences, tokenizer.word_index

# Padding sequences
def pad_sequences(sequences, max_seq_length):
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    return padded_sequences

# Preprocess data
texts = [conversation['body'] for conversation in data]
sequences, word_to_index = tokenize_texts(texts)
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
    seq2seq_model.load_weights('model_weights.h5')
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
    indices = tf.range(len(input_data))
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_input_data = tf.gather(input_data, shuffled_indices)
    shuffled_target_data = tf.gather(target_data, shuffled_indices)
    
    # Iterate over batches
    for i in range(0, len(input_data), batch_size):
        batch_input = shuffled_input_data[i:i+batch_size]
        batch_target = shuffled_target_data[i:i+batch_size]
        
        # Forward pass
        with tf.GradientTape() as tape:
            predictions = seq2seq_model(batch_input, batch_input, training=True)
            loss = loss_function(batch_target, predictions)
        
        # Backpropagation
        gradients = tape.gradient(loss, seq2seq_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, seq2seq_model.trainable_variables))
        
        # Print loss for monitoring
        print(f"Batch {i//batch_size + 1}/{len(input_data)//batch_size}, Loss: {loss.numpy()}")

# Save the model weights
seq2seq_model.save_weights('model_weights.h5')
print("Model weights saved successfully...")
