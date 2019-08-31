from aane import AANE
from helpers import read_graph, read_features, embedding_saver, parameter_parser

def learn_model(args):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    A = read_graph(args.edge_path)
    X = read_features(args.feature_path)
    H = AANE(A, X, 128).function()
    embedding_saver(H, args.output_path)
    
if __name__ == "__main__":
    args = parameter_parser()
    learn_model(args)
