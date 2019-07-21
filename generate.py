from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
import io
import json

model = load_model("IR_podcast_v3.h5")

tokenizer = Tokenizer()
seed_text = "I once knew a man named Justin"
next_words = 50

# Load Podcast vocab
tokenizer = Tokenizer()
# All the IR podcast transcripts from https://building.infinite.red/
data = open('./podcasts.txt').read()

# break it into lines
corpus = data.lower().split("\n")

# converts each word into a unique number (ordered by most used)
tokenizer.fit_on_texts(corpus)
# print(tokenizer.word_index)
# exit()
total_words = len(tokenizer.word_index) + 1

# Gathered from other scripts
max_sequence_len = 252

# Generate `next_words` using the model
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding='pre')
    print(token_list)
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
