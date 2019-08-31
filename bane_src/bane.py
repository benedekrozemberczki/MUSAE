import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD

class BANE(object):
    """
    Binarized Attributed Network Embedding Class (ICDM 2018).
    """
    def __init__(self, args, P, X):
        """
        Setting up a BANE model.
        :param args: Arguments object.
        :param P: Normalized connectivity matrix.
        :param X: Feature matrix.
        """
        self.args = args
        self.P = P
        self.X = X

    def fit(self):
        """
        Creating a BANE embedding.
        1. Running SVD.
        2. Running power iterations and CDC.
        """

        print("\nFitting BANE model.\nBase SVD fitting started.")
        
        self.fit_base_SVD_model()

        print("SVD completed.\nFitting binary model.\n")

        self.binary_optimize()


    def fit_base_SVD_model(self):
        """
        Reducing the dimensionality with SVD in the 1st step.
        """
        self.P = self.P.dot(self.X)
        self.model = TruncatedSVD(n_components=self.args.dimensions, n_iter = 10)
        self.model.fit(self.P)
        self.P = self.model.fit_transform(self.P)


    def update_G(self):
        """
        Uopdating the kernel matrix.
        """

        self.G = inv(np.dot(self.B.transpose(), self.B)+self.args.alpha*np.eye(self.args.dimensions)).dot(self.B.transpose()).dot(self.P)

    def update_Q(self):
        """
        Updating the rescaled target matrix.
        """
        self.Q = self.G.dot(self.P.transpose()).transpose()

    def update_B(self):
        """
        Updating the embedding matrix.
        """
        for i in tqdm(range(self.args.approximation_rounds), desc="Inner approximation:"):
            for dimension in range(self.args.dimensions):
                selector = [x for x in range(self.args.dimensions) if x != dimension]
                self.B[:,dimension] = np.sign(self.Q[:,dimension]-self.B[:,selector].dot(self.G[selector,:]).dot(self.G[:,dimension]).transpose())

    def binary_optimize(self):
        """
        Starting 2nd optimization phase with power iterations and CDC.
        """

        self.B = np.sign(np.random.normal(size=(self.P.shape[0], self.args.dimensions)))

        for iteration in tqdm(range(self.args.binarization_rounds), desc="Power iteration", leave=True):
            self.update_G()
            self.update_Q()
            self.update_B()

    def save_embedding(self):
        """
        Saving the embedding.
        """
        self.out = np.concatenate([np.array(range(self.B.shape[0])).reshape(-1,1),self.B],axis=1)
        self.out = pd.DataFrame(self.out,columns = ["id"] + [ "x_"+str(dim) for dim in range(self.args.dimensions)])
        self.out.to_csv(self.args.output_path, index = None)
        print("\n\nModel saved.")
