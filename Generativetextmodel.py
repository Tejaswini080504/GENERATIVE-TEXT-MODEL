!pip install tensorflow transformers torch
!pip install --upgrade tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
# Load GPT2 pre-trained model and tokenizer
model_name = 'gpt2'  # You can also try 'gpt2-medium' or 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate text using GPT-2
def generate_text(prompt, max_length=100):
    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate output sequence
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# User input and text generation
prompt = input("Enter a prompt for GPT-2 to generate text: ")
generated_paragraph = generate_text(prompt, max_length=200)
print(f"\nGenerated Text:\n{generated_paragraph}")
# Example text data - Replace with your own dataset
text_data = """
Once upon a time in a land far, far away, there lived a wise king. He ruled the kingdom with justice and compassion.
The people loved him for his fairness, and the kingdom flourished. However, dark times loomed on the horizon.
An enemy army approached from the north, threatening to take over the land. The king knew he had to act quickly.
He gathered his advisors and formulated a plan. The battle was fierce, but in the end, the king's strategy led to victory.
"""

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = []
for line in text_data.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        sequence = token_list[:i+1]
        sequences.append(sequence)

# Padding sequences
max_sequence_length = max([len(x) for x in sequences])
X = np.array([pad_sequences([seq], maxlen=max_sequence_length)[0] for seq in sequences])
y = tf.keras.utils.to_categorical(X[:, -1], num_classes=total_words)

# Prepare features and labels for LSTM
X = X[:, :-1]
# LSTM Model Architecture
model_lstm = Sequential()
model_lstm.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
model_lstm.add(LSTM(150, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(150))
model_lstm.add(Dense(total_words, activation='softmax'))

# Compile the model
model_lstm.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

# Train the LSTM model (This is a small dataset, so it might take a while on larger datasets)
model_lstm.fit(X, y, epochs=50, batch_size=128)
# Function to generate text using LSTM
def generate_text_lstm(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1)
        
        # Predict the next word
        predicted_prob = model_lstm.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_prob)
        
        # Convert the index back to the word
        predicted_word = tokenizer.index_word[predicted_word_index]
        
        # Append the word to the seed text
        seed_text += " " + predicted_word
    return seed_text

# Generate text from LSTM model
seed_text = input("Enter a seed text for LSTM to generate: ")
generated_text_lstm = generate_text_lstm(seed_text, next_words=50)
print(f"\nGenerated Text (LSTM):\n{generated_text_lstm}")
