"""Running MUSAE."""

from musae import MUSAE
from utils import tab_printer
from parser import parameter_parser

def main(args):
    """
    Multi-scale attributed node embedding machine calling wrapper.
    :param args: Arguments object parsed up.
    """
    model = MUSAE(args)
    model.do_sampling()
    model.learn_embedding()
    model.save_embedding()
    model.save_logs()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    main(args)
