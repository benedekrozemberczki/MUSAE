from musae import MUSAE
from utils import tab_printer
from parser import parameter_parser
import time

def main(args):
    """
    Multi-scale attributed node embedding machine calling wrapper.
    :param args: Arguments object parsed up.
    """
    model = MUSAE(args)
    model.do_sampling()
    print(args.graph_input)
    print("MUSAE")
    for core in [1,2,4,8]:
        print("CORES: ",core)
        for dimension in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]:
            args.dimensions = dimension
            args.model = "musae"
            model.args = args
            start = time.time()
            model.learn_embedding()
            end = time.time()
            print("("+ str(dimension)+","+str(round(10*(end-start),4))+")")
    print("AE")
    for core in [1,2,4,8]:
        print("CORES: ",core)
        for dimension in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]:
            args.dimensions = dimension
            args.model = "ae"
            args.workers = core
            model.args = args
            start = time.time()
            model.learn_embedding()
            end = time.time()
            print("("+ str(dimension)+","+str(round(10*(end-start),4))+")")
    #model.save_embedding()
    #model.save_logs()

if __name__ == "__main__":
    args = parameter_parser()
    args.graph_input = "./input/edges/chameleon_edges.csv"
    args.features_input = "./input/features/chameleon_features.json"
    main(args)
    args.graph_input = "./input/edges/chameleon_edges.csv"
    args.features_input ="./input/features/chameleon_features.json"
    main(args)
    args.graph_input = "./input/edges/chameleon_edges.csv"
    args.features_input ="./input/features/chameleon_features.json"
    main(args)
