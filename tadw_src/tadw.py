import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
from numpy.linalg import inv
from numpy import linalg as la
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from sklearn.preprocessing import normalize
class TADW(object):

    def __init__(self,A, X, args):
        self.args = args
        self.lamb = self.args.lambd
        self.M = A
        self.X = X.T
        self.dim = self.args.dimensions
        self.create_features()

    def create_features(self):
        svd = TruncatedSVD(n_components=200, n_iter=50)
        svd.fit(self.X)
        self.T = svd.transform(self.X).T



    def train(self):
        self.node_size = self.M.shape[0]
        self.feature_size = self.T.shape[0]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update

        for i in tqdm(range(self.args.iterations)):

            B = np.dot(self.H, self.T)
    
            execs = (self.M.T.dot(B.T)).T
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - 2*execs + self.lamb*self.W
            Hess = 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            drv = np.reshape(drv, [self.dim*self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim*self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [self.dim*self.node_size, 1])

                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecW = vecW + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            execs = (self.M.T.dot(self.W.T)).T
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)            -execs ), self.T.T) + self.lamb*self.H
            drv = np.reshape(drv, (self.dim*self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim*self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))      + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
        self.Vecs = np.hstack((normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
        
    def save_embedding(self):
        """
        Saving the embedding on disk.
        """
        print("\nSaving the embedding.\n")
        columns = ["id"] + ["X_"+str(dim) for dim in range(2*self.args.dimensions)]
        ids = np.array(range(0,self.M.shape[0])).reshape(-1,1)
        self.W = np.concatenate([ids, self.Vecs], axis = 1)
        self.out = pd.DataFrame(self.W, columns = columns)
        self.out.to_csv(self.args.output_path, index = None)

        
        
