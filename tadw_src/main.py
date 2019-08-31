from tadw import TADW
from helpers import parameter_parser, read_graph, read_sparse_features, tab_printer

def learn_model(args):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    A = read_graph(args.edge_path, args.order)
    X = read_sparse_features(args.feature_path)

    model = TADW(A, X, args)
    model.train()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    learn_model(args)
