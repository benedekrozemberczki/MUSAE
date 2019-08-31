from tene import TENE
from helpers import parameter_parser, read_graph, read_features, read_sparse_features, tab_printer

def learn_model(args):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    X = read_graph(args.edge_path, args.order)
    if args.features == "dense":
        T = read_features(args.feature_path)
    elif args.features == "sparse":
        T = read_sparse_features(args.feature_path)
    model = TENE(X, T, args)
    model.optimize()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    learn_model(args)
