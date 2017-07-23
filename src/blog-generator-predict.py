# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import codecs 
path = get_file('sentense.txt', origin='sentense.txt')
text = codecs.open(path, 'r', 'utf-8').read().lower()
fout = codecs.open('results/result.txt', 'a', 'utf-8')

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
model = load_model('models/model.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#start_index = random.randint(0, len(text) - maxlen - 1)
start_index = 0

for diversity in [0.2, 0.5, 1.0, 1.2]:

    generated = ''
    #sentence = text[start_index: start_index + maxlen]
    #sentence = 'laravelの開発は'
    #sentence = 'こんにちは、yamatoです。'
    sentence = 'dockerとは'
    generated += sentence
    fout.write("\n****************************\n")
    fout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        fout.write(next_char)
