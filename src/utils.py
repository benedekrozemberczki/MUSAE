import json
import random
import pandas as pd
import networkx as nx
from texttable import Texttable
from gensim.models.doc2vec import TaggedDocument

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def load_graph(graph_path):
    """
    """
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]),int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

def load_features(features_path):
    """
    """
    features = json.load(open(features_path))
    features = {str(k): [str(val) for val in v] for k,v in features.items()}
    return features

def create_documents(features):
    """
    """
    docs = [TaggedDocument(words = v, tags = [str(k)]) for k, v in features.items()]
    return docs
