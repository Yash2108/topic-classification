import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from preprocess import Preprocessor

processor=Preprocessor()
sentence=input("Enter a sentence: ")
embedding=processor.get_embedding(processor.apply(sentence))
G=nx.read_adjlist('weights/graph.adjlist')
labelled_embeddings = [i for i in np.load('data/labelled_embeddings.npy')]
edge_sim_labelled=[]
for i in G.nodes:
    cosine_sim_val=cosine_similarity( [ labelled_embeddings[int(i)] ], [ embedding ])
    if cosine_sim_val>0.8:
        edge_sim_labelled.append([i, len(G.nodes)])

G.add_node(len(G.nodes))
G.add_edges_from(edge_sim_labelled)
print("Nodes similar to the input sentence:", [i[0] for i in edge_sim_labelled])