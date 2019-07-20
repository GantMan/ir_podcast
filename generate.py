from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np 

model = load_model("IR_podcast_v3.h5")

tokenizer = Tokenizer()
seed_text = "I think Todd is"
next_words = 50

# Load Podcast vocab
tokenizer = Tokenizer()
# All the IR podcast transcripts from https://building.infinite.red/
data = open('./podcasts.txt').read()

# break it into lines
corpus = data.lower().split("\n")

# converts each word into a unique number (ordered by most used)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences ( so they are all the same length )
max_sequence_len = max([len(x) for x in input_sequences])

# Generate `next_words` using the model
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len -1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)