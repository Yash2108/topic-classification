from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import F1Score
from nltk.corpus import stopwords
from nltk import download
import numpy as np
download('stopwords')

import re
from sentence_transformers import SentenceTransformer


class Preprocessor:
  '''
    Removes punctuations, numbers, whitespaces
    Converts sentences into lower case
  '''

  def __init__(self, numbers=True, punct=True, stopwords=True, empty_sentence=True, model='bert-base-nli-mean-tokens'):
    self.numbers = numbers
    self.punct = punct
    self.stopwords = stopwords
    self.empty_sentence = empty_sentence
    self.model = SentenceTransformer(model)

  def apply(self, sentence):
    sentence = sentence.lower()
    if self.numbers:
      sentence = Preprocessor.remove_numbers(sentence)

    if self.punct: 
      sentence = Preprocessor.remove_punct(sentence)

    if self.stopwords: 
      sentence = Preprocessor.remove_stopwords(sentence)

    if self.empty_sentence: 
      sentence = Preprocessor.empty_sentence(sentence)

    if sentence!=None:
        sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
  
  @staticmethod
  def remove_punct(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence
  
  @staticmethod
  def remove_numbers(sentence):
    sentence = re.sub(r'[0-9]', '', sentence)
    return sentence

  @staticmethod
  def remove_stopwords(sentence):
    sentence_clean = ' '.join( [ word for word in sentence.split() if word.lower() not in set( stopwords.words('english') ) ] )
    return sentence_clean

  
  @staticmethod
  def empty_sentence(sentence):
    words=sentence.split()
    if (not all(elem == "" for elem in sentence)) and len(sentence)>2:
        return sentence
    else:
        return None

  def get_embedding(self, sentence):
      return self.model.encode(sentence)
processor=Preprocessor()

# Example sentences to try:
# We look forward this year to completing our compact of free association agreement with the United States, which will reflect that close and special relationship.
# We hope that the projected quadripartite, high-level talks between Turkey, Greece and the leaders of the two Cypriot communities will still materialise in the near future and bring us closer to a settlement based on the concept of a bicommunal and bizonal federation.
# The newly formed Government of National Unity has embarked on that task with a renewed plan of action and a number of priority objectives and benchmarks.
# In our age, the power of human rights has become global and cannot serve any particular interests.

sentence=input("Enter a sentence: ")
embedding=processor.get_embedding(processor.apply(sentence))

labels_enc = {'democracy':0, 'development':1, 'greeting':2, 'human':3, 'security':4, 'un':5}
labels_dec = {0:'democracy', 1:'development', 2:'greeting', 3:'human', 4:'security', 5:'un'}

optimizer = Adam(1e-5)

inputs = Input(shape=(768,))
hidden_layer = Dense(256, activation='relu')(inputs)
dropout1 = Dropout(0.3)(hidden_layer)
hidden_layer2 = Dense(128, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(hidden_layer2)
hidden_layer3 = Dense(64, activation='relu')(dropout2)
hidden_layer4 = Dense(32, activation='relu')(hidden_layer3)
outputs = Dense(6, activation='softmax')(hidden_layer4)
model_nn = Model(inputs=inputs, outputs=outputs)

model_nn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', F1Score()])

model_nn.load_weights('weights/second_classifier.weights.h5')
embedding=np.reshape(embedding, (1, 768))
predicted_label=model_nn.predict(embedding)
print(labels_dec[np.argmax(predicted_label)])