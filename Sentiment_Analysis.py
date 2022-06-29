from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import accuracy_score
import os
from time import time
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.initializers import Constant
from keras.models import Sequential
from tqdm import tqdm
from tensorflow.keras.regularizers import l1
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import string
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
stop = set(stopwords.words('english'))
# import gensim

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.iloc[:, :2]
data['message_len'] = data.v2.apply(len)
data['v1'] = data['v1'].replace({'ham': 0, 'spam': 1})

plt.figure(figsize=(12, 8))

data[data.v1 == 0].message_len.plot(bins=35, kind='hist', color='blue',
                                    label='Ham messages', alpha=0.6)
data[data.v1 == 1].message_len.plot(kind='hist', color='red',
                                    label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")

df = data.iloc[:, :2]


def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split(' ')

    text = [w for w in text if not w in set(stopwords.words('english'))]

    text = ' '.join(text)

    return text


df['v2'] = df['v2'].apply(lambda x: clean_text(x))


def create_corpus(df):
    corpus = []
    for tweet in tqdm(df['v2']):
        words = [word.lower() for word in word_tokenize(
            tweet) if((word.isalpha() == 1) & (word not in stop))]
        corpus.append(words)
    return corpus


corpus = create_corpus(df)

embedding_dict = {}
with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors
f.close()

MAX_LEN = 10
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)

tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN,
                          truncating='post', padding='post')
word_index = tokenizer_obj.word_index
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, 50))

for word, i in tqdm(word_index.items()):
    if i > num_words:
        continue

    emb_vec = embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec

X_train, X_val, y_train, y_val = train_test_split(
    tweet_pad, df.v1, test_size=.2, random_state=2)

model = Sequential()

embedding_layer = Embedding(num_words, 50, embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_LEN, trainable=False)

model.add(embedding_layer)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(tf.keras.layers.LSTM(32, return_sequences=True))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimzer = Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=32, epochs=10,
                    validation_data=(X_val, y_val), verbose=1)
model_loss = pd.DataFrame(model.history.history)

model_loss[['loss', 'val_loss']].plot(ylim=[0, 1])
model_loss[['acc', 'val_acc']].plot(ylim=[0, 1])
