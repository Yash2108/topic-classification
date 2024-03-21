import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow import one_hot

from tensorflow.keras.metrics import F1Score

from gensim.models import Word2Vec
from node2vec import Node2Vec
import networkx as nx


# def convert_to_embeddings(emb):
#    ''' Function to read numpy embeddings'''
#    return np.array([ [float(j) for j in i.strip().split()] for i in emb[1:-1].split('\n') ]).reshape(-1)

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

# Generate graph and node embeddings
def make_graph(data, cos_sim=0.9, plot_graph=False):
    edges_subset=list(combinations(data.index, 2))
    edge_sim_labelled=[]

    for idx1, idx2 in tqdm(edges_subset):

        cosine_sim_val=cosine_similarity( [ data['embeddings'][idx1] ], [ data['embeddings'][idx2] ])

        if cosine_sim_val>cos_sim:
            edge_sim_labelled.append([idx1, idx2])
    
    G = nx.Graph()
    G.add_nodes_from(data.index)
    G.add_edges_from(edge_sim_labelled)

    if plot_graph:
        plt.figure(figsize=(20,20))
        nx.draw(G)
    return G

def make_N2V_embeddings(G, cos_sim=0.9, embedding_size=512, walk_length=10, num_walks=80, window=10, connect_type='sim_country_year'):
    filename=f"emb{embedding_size}_wklen{walk_length}_nwalk{num_walks}_win{window}_{connect_type}"
    if connect_type.find('sim'):
        filename+=f"_csim{cos_sim}"
    filename+=f'_GLen{len(G.nodes)}'
    node2vec = Node2Vec(G, dimensions=embedding_size, walk_length=walk_length,num_walks=num_walks, workers=4)
    node2vec_model = node2vec.fit(window=window, min_count=1,batch_words=4)

    embeddings_map = node2vec_model.wv
    embeddings = embeddings_map[[i for i in range(len(G.nodes))]]

    node2vec_model.save('weights/'+filename+'.model')

    return embeddings

def load_N2V_embeddings(G_len, embedding_path='weights/emb512_wklen30_nwalk160_win20_cossim0.9_sim.model'):
    node2vec_model=Word2Vec.load(embedding_path)
    embeddings = node2vec_model.wv[[i for i in range(G_len)]]
    return embeddings

embedding_size=512
walk_length=30
num_walks=160
window=20
connect_type='sim'
G=make_graph(labelled_cleaned, cos_sim=0.9, plot_graph=False)
nx.write_adjlist(G, 'weights/graph.adjlist')
embeddings=make_N2V_embeddings(G, embedding_size=embedding_size, walk_length=walk_length, num_walks=num_walks, window=window, connect_type=connect_type)


labels={'security': 0, 
        'greeting': 1, 
        'development': 2, 
        'democracy': 3, 
        'human': 4, 
        'un': 5}

train_labels_one_hot=one_hot(
    [ labels[i] for i in labelled_cleaned['coding'].values ],
    len(labels)).numpy()

X_train, X_eval, y_train, y_eval = train_test_split(embeddings, train_labels_one_hot)

# Neural Network

class Classifier(Model):
    def __init__(self, hidden_dims, output_size, dropout_rate):
        super(Classifier, self).__init__()
        self.dense1 = Dense(hidden_dims[0], activation='relu')
        self.dense2 = Dense(hidden_dims[1], activation='relu')
        self.dense3 = Dense(output_size, activation='softmax')
        self.bnorm1 = BatchNormalization()
        self.bnorm2 = BatchNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.bnorm1(x)
        
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.bnorm2(x)
        
        output = self.dense3(x)
        return output

tf_model = Classifier((256, 128), len(labels), 0.3)
tf_model.build(input_shape=(None, embedding_size))
tf_model.compile(optimizer=Adagrad(learning_rate=0.001),
              loss=CategoricalCrossentropy(),
              metrics=[F1Score()])
tf_model.summary()

his = tf_model.fit(X_train, y_train, validation_data=(X_eval, y_eval), epochs=400, verbose=1, 
                           callbacks = EarlyStopping(monitor='loss', patience=10))

tf_model.save_weights('weights/graph_approach.weights.h5')

