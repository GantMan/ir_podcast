import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import tensorflow.keras.utils as ku 
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np 

weights_file = "weights.best_ir_porcast.hdf5"

print ('Loading Model')
model = load_model("IR_podcast.h5")

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

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


# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
# Each sentence builds to the next word
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

label = ku.to_categorical(label, num_classes=total_words)        

# About 1hr per epoch on single GPU
history = model.fit(predictors, label, epochs=12, verbose=1, callbacks=[ModelCheckpoint(
        weights_file, monitor='acc', verbose=1, save_best_only=True, mode='max')])

# Graph the history of the training
import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()


# Save it for later
print('Saving Model')
model.save("IR_podcast.h5")