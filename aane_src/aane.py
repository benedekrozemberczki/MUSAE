import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import ceil
from tqdm import tqdm

class AANE:
    """Jointly embed Net and Attri into embedding representation H
    H = AANE(Net,Attri,d).function()
    H = AANE(Net,Attri,d,lambd,rho).function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter).function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter,'Att').function()
    H = AANE(Net,Attri,d,lambd,rho,maxiter,'Att',splitnum).function()
    :param Net: the weighted adjacency matrix
    :param Attri: the attribute information matrix with row denotes nodes
    :param d: the dimension of the embedding representation
    :param lambd: the regularization parameter
    :param rho: the penalty parameter
    :param maxiter: the maximum number of iteration
    :param 'Att': refers to conduct Initialization from the SVD of Attri
    :param splitnum: the number of pieces we split the SA for limited cache
    :return: the embedding representation H
    Copyright 2017 & 2018, Xiao Huang and Jundong Li.
    $Revision: 1.0.2 $  $Date: 2018/02/19 00:00:00 $
    """
    def __init__(self, Net, Attri, d, *varargs):
        self.maxiter = 20  # Max num of iteration
        [self.n, m] = Attri.shape  # n = Total num of nodes, m = attribute category num
        self.lambd = 0.05  # Initial regularization parameter
        self.rho = 5  # Initial penalty parameter
        splitnum = 1  # number of pieces we split the SA for limited cache
        if len(varargs) >= 4 and varargs[3] == 'Att':
            sumcol = np.arange(m)
            np.random.shuffle(sumcol)
            self.H = svds(Attri[:, sumcol[0:min(10 * d, m)]], d)[0]
        else:
            sumcol = Net.sum(0)
            self.H = svds(Net[:, sorted(range(self.n), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * d, self.n)]], d)[0]

        if len(varargs) > 0:
            self.lambd = varargs[0]
            self.rho = varargs[1]
            if len(varargs) >= 3:
                self.maxiter = varargs[2]
                if len(varargs) >= 5:
                    splitnum = varargs[4]
        self.block = min(int(ceil(float(self.n) / splitnum)), 7575)  # Treat at least each 7575 nodes as a block
        self.splitnum = int(ceil(float(self.n) / self.block))
        with np.errstate(divide='ignore'):  # inf will be ignored
            self.Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))
        self.Z = self.H.copy()
        self.affi = -1  # Index for affinity matrix sa
        self.U = np.zeros((self.n, d))
        self.nexidx = np.split(Net.indices, Net.indptr[1:-1])
        self.Net = np.split(Net.data, Net.indptr[1:-1])
        self.d = d


    '''################# Update functions #################'''
    def updateH(self):
        xtx = np.dot(self.Z.transpose(), self.Z) * 2 + self.rho * np.eye(self.d)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.Z) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.Z[self.nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.H[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                        self.H[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                                                                   self.Z[i, :] - self.U[i, :]))
                    else:
                        self.H[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (
                                    self.Z[i, :] - self.U[i, :]))
    def updateZ(self):
        xtx = np.dot(self.H.transpose(), self.H) * 2 + self.rho * np.eye(self.d)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                self.sa = self.Attri[:, range(indexblock, indexblock + min(self.n - indexblock, self.block))].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.H) * 2
            for i in range(indexblock, indexblock + min(self.n - indexblock, self.block)):
                neighbor = self.H[self.nexidx[i], :]  # the set of adjacent nodes of node i
                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.Z[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index
                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Net[i][nzidx]) / normi_j[nzidx]
                        self.Z[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.d), sums[i - indexblock, :] + (
                                    neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (
                                                                   self.H[i, :] + self.U[i, :]))
                    else:
                        self.Z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (
                                    self.H[i, :] + self.U[i, :]))

    def function(self):
        self.updateH()
        '''################# Iterations #################'''
        for __ in tqdm(range(self.maxiter - 1)):
            self.updateZ()
            self.U = self.U + self.H - self.Z
            self.updateH()
        return self.H


