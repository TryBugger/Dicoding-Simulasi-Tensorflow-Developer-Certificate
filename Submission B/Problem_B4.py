# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
import re, string, nltk, contractions
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    def text_cleaning(text):
        text = text.lower()
        text = contractions.fix(text)
        punct = string.punctuation

        cleaned_x = ''
        for char in text:
            if char not in punct:
                cleaned_x = cleaned_x + char

        return cleaned_x

    def tokenize(text):
        text = text.split(' ')

        while '' in text:
            text.remove('')

        return text

    stop_words = list(stopwords.words('english'))

    def remove_stopword(text):
        filtered_sentence = []
        for word in text:
            if word not in stop_words:
                filtered_sentence.append(word)

        return filtered_sentence

    def pos_tagging(text):
        tag = nltk.pos_tag([text])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)


    def lemmatization(text):
        lemma = WordNetLemmatizer()
        text = [lemma.lemmatize(word, pos_tagging(word)) for word in text]

        return text

    def join_string(text):
        return ' '.join(text)

    bbc['preprocessed_text'] = bbc['text'].map(lambda x: text_cleaning(x))
    bbc['preprocessed_text'] = bbc['preprocessed_text'].map(lambda x: tokenize(x))
    bbc['preprocessed_text'] = bbc['preprocessed_text'].map(lambda x: remove_stopword(x))
    # bbc['preprocessed_text'] = bbc['preprocessed_text'].map(lambda x: lemmatization(x))
    bbc['preprocessed_text'] = bbc['preprocessed_text'].map(lambda x: join_string(x))

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    # enc = OrdinalEncoder()
    # encoded_category = enc.fit_transform(np.array(bbc['category']).reshape(-1, 1)).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(bbc['preprocessed_text'], np.array(bbc['category']), shuffle=False,
                                                        train_size=training_portion)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # YOUR CODE HERE
    tokenizer.fit_on_texts(X_train)

    training_sequences = tokenizer.texts_to_sequences(X_train)
    training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                    truncating=trunc_type, padding=padding_type)
    val_sequences = tokenizer.texts_to_sequences(X_test)
    val_padded = pad_sequences(val_sequences, maxlen=max_length)

    tokenizer = Tokenizer()  # YOUR CODE HERE
    tokenizer.fit_on_texts(y_train)

    train_label = np.array(tokenizer.texts_to_sequences(y_train))
    val_label = np.array(tokenizer.texts_to_sequences(y_test))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(0.0012)),
        # tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L2(l1=0.001, l2=0.001)),

        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics='acc')

    class stop_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > .93) and (logs.get('val_acc') > .93):
                self.model.stop_training = True

    model.fit(training_padded, train_label,
              epochs=150,
              validation_data=(val_padded, val_label),
              callbacks=[stop_callback()])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
