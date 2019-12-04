import pandas as pd
import re
import numpy as np
import keras
import pydot
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Input
from keras.layers import LSTM, Lambda,Dropout, Dense, Embedding, Flatten, Conv1D
from keras.layers.merge import add
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb':   'data/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
# print(df.iloc[:3])

max_len = 50
batch_size = 32
Epoach = 10

Train = df.sentence.values
Label = df.label.values


def ELMO_Classification(Train, Label):
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

    def Balance_batch(Data, Label, batch_size=32):
        Data = np.array(Data)
        m = len(Data) % batch_size
        if m != 0:
            x = np.random.randint(low=0, high=len(Data), size=batch_size - m)
            Data = np.concatenate((Data, Data[x]))
            Label = np.array(Label)
            Label = np.concatenate((Label, Label[x]))
        assert len(Label) == len(Data), print('lenght of data and labels are not equal!!')
        return Data, Label

    def ElmoEmbedding(x):
        return elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(batch_size * [max_len])
        },

            signature='tokens',
            as_dict=True)["elmo"]

    Train = [text_to_word_sequence(s.lower()) for s in Train]
    Train = pad_sequences(sequences=Train, padding='post', maxlen=max_len, dtype=object, value="__PAD__")
    x_train, x_test, y_train, y_test = train_test_split(Train, Label, test_size=0.20, random_state=42)
    x_train, y_train = Balance_batch(x_train, y_train, batch_size=batch_size)
    x_test, y_test = Balance_batch(x_test, y_test, batch_size=batch_size)
    def build_model():
        Inputs = Input(shape=(max_len,), dtype=tf.string)
        embedding_layer = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(Inputs)
        # If you want to use Convolutional networks leave 4 lines below uncommented

        cnn = Conv1D(128,5,activation='relu')(embedding_layer)
        cnn = Conv1D(128,5,activation='relu')(cnn)
        cnn = Flatten()(cnn)
        dense = Dense(128, activation='relu')(cnn)

        # If you want to use LSTM make three above lines commented and do uncomment three lines below

        # lstm = LSTM(128,return_sequences=True )(embedding_layer)
        # lstm = LSTM(128, return_sequences=False)(lstm)
        # dense = Dense(128, activation='relu')(lstm)

        dense = Dense(1, activation='sigmoid')(dense)
        
        model = Model(input=[Inputs], outputs=dense)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        plot_model(model, to_file='Elmo_plot4a.png', show_shapes=True, show_layer_names=True)
        print(model.summary())

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=Epoach)
        return history, model

    history, model = build_model()
    score = model.evaluate(x_test, y_test, verbose=0)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])


ELMO_Classification(Train,Label)



