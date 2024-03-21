import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
from tqdm import tqdm
from preprocess import Preprocessor

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import expand_dims
from tensorflow.keras.metrics import F1Score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Loading labelled data and embeddings
labelled_cleaned = pd.read_csv('data/labelled_cleaned.csv')
labelled_cleaned.drop(['Unnamed: 0'], inplace=True, axis=1)
labelled_embeddings = [i for i in np.load('data/labelled_embeddings.npy')]
labelled_cleaned['embeddings'] = [i for i in labelled_embeddings]

# Loading unlabelled data and embeddings
unlabelled_cleaned = pd.read_csv('data/unlabelled_cleaned.csv')
unlabelled_cleaned.drop(['Unnamed: 0'], inplace=True, axis=1)
unlabelled_embeddings = [i for i in np.load('data/unlabelled_embeddings_2k.npy')]
unlabelled_cleaned['embeddings'] = [i for i in unlabelled_embeddings]


labels_enc = {'democracy':0, 'development':1, 'greeting':2, 'human':3, 'security':4, 'un':5}
labels_dec = {0:'democracy', 1:'development', 2:'greeting', 3:'human', 4:'security', 5:'un'}
labelled_cleaned['coding']=labelled_cleaned['coding'].replace(labels_enc)
labelled_cleaned['coding'].values

# Classifier 1
input_shape=768
output_shape=6

X_train = np.stack(labelled_cleaned['embeddings'].tolist())
y_train = to_categorical(labelled_cleaned['coding'].values, num_classes=output_shape)

X_train_split, X_eval, y_train_split, y_eval = train_test_split(X_train, y_train, test_size=0.05)
oversample = SMOTE()
X_train_split, y_train_split = oversample.fit_resample(X_train_split, y_train_split)

inputs = Input(shape=(input_shape,))
hidden_layer = Dense(64, activation='relu')(inputs)
hidden_layer2 = Dense(32, activation='relu')(hidden_layer)
outputs = Dense(output_shape, activation='softmax')(hidden_layer2)
model_nn = Model(inputs=inputs, outputs=outputs)
model_nn.summary()
optimizer = Adam(1e-5)

model_nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', F1Score()])

history = model_nn.fit(X_train_split, y_train_split, epochs=500, batch_size=16, validation_data=(X_eval, y_eval) )

# Saving model weights
model_nn.save_weights('weights/first_classifier.weights.h5')
unlabelled = unlabelled_cleaned.copy()

batch_size = 20000
unlabelled_X = list(unlabelled['text_clean'][0:batch_size])

# Generating pseudo labels for 20k rows
psuedo_preds = []
processor = Preprocessor()
for sentence in tqdm(unlabelled_X):
  encoded_sent = processor.get_embedding(sentence)
  psuedo_preds.append(labels_dec[np.argmax(model_nn.predict(expand_dims(encoded_sent, axis=0)))])

pd.DataFrame([unlabelled_X, psuedo_preds]).to_csv('data/unlabelled_predictions_smote.csv')

out_data = pd.read_csv('data/unlabelled_predictions_smote.csv')

out_data = out_data.transpose()
out_data.columns = ['text_clean', 'coding']
out_data.to_csv('data/psuedo_labelled_data.csv')

labelled_sliced = labelled_cleaned[['text_clean', 'coding']]
out_data = out_data[['text_clean', 'coding']][1:]
labelled_stacked = pd.concat([labelled_sliced, out_data])
labelled_stacked = labelled_stacked.reset_index(drop=True)
labelled_stacked.to_csv('data/labelled_stacked.csv')
labelled_stacked = pd.read_csv('data/labelled_stacked.csv')
X_train = np.array([processor.get_embedding(i) for i in labelled_stacked['text_clean']])
np.save('data/combined_embeddings.npy', X_train)

X_train= np.load('data/combined_embeddings.npy')

labelled_stacked['coding']=labelled_stacked['coding'].replace(labels_enc)
y_train = to_categorical(labelled_stacked['coding'].values, num_classes=output_shape)


# Classifier 2
optimizer = Adam(1e-5)

inputs = Input(shape=(input_shape,))
hidden_layer = Dense(256, activation='relu')(inputs)
dropout1 = Dropout(0.3)(hidden_layer)
hidden_layer2 = Dense(128, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(hidden_layer2)
hidden_layer3 = Dense(64, activation='relu')(dropout2)
hidden_layer4 = Dense(32, activation='relu')(hidden_layer3)
outputs = Dense(output_shape, activation='softmax')(hidden_layer4)
model_nn = Model(inputs=inputs, outputs=outputs)

model_nn.summary()

model_nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', F1Score()])

X_train_split, X_eval, y_train_split, y_eval = train_test_split(X_train, y_train, test_size=0.05)

history = model_nn.fit(X_train_split, y_train_split, epochs=100, batch_size=32, validation_data=(X_eval, y_eval))

model_nn.save_weights('weights/second_classifier.weights.h5')
