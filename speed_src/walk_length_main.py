from musae import MUSAE
from utils import tab_printer
from parser import parameter_parser
import time

def main(args):
    """
    Multi-scale attributed node embedding machine calling wrapper.
    :param args: Arguments object parsed up.
    """
    print(args.graph_input)
    print("MUSAE")
    for core in [1,2,4,8]:
        print("CORES: ",core)
        for length in [20,40,60,80,100,120,140,160]:
            args.workers = core
            args.walk_length = length
            args.model = "musae"
            model = MUSAE(args)
            model.do_sampling()
            start = time.time()
            model.learn_embedding()
            end = time.time()
            print("("+ str(length)+","+str(round((end-start),4))+")")
    print("AE")
    args.dimensions = 64
    for core in [1,2,4,8]:
        print("CORES: ",core)
        for length in [20,40,60,80,100,120,140,160]:
            args.workers = core
            args.walk_length = length
            args.model = "musae"
            model = MUSAE(args)
            model.do_sampling()
            start = time.time()
            model.learn_embedding()
            end = time.time()
            print("("+ str(length)+","+str(round((end-start),4))+")")
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
