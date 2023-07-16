# cao bin, HKUST, China
# free to charge for academic communication

import numpy as np
import scipy.linalg
from sklearn.gaussian_process.kernels import  RBF 

class TCA():
    def __init__(self, dim=30, lamda=1, gamma=1):
        '''
        :param dim: data dimension after projection
        :param lamb: lambda value, Lagrange multiplier
        :param gamma: length scale for rbf kernel
        '''
        self.dim = dim
        self.lamda = lamda
        self.kernel = 0.5*RBF(gamma,"fixed")

    def fit(self, Xs, Xt, ):
        '''
        :param Xs: ns * m_feature, source domain data 
        :param Xt: nt * m_feature, target domain data
        Projecting Xs and Xt to a lower dimension by TCA
        source/target domain data expressed in a mapping space
        :return: Xs_new and Xt_new 
        '''
        # formular in paper Domain Adaptation via Transfer Component Analysis
        # Eq.(2) 
        X = np.vstack((Xs, Xt))
        K = self.kernel(X)
        # cal matrix L 
        ns, nt = len(Xs), len(Xt)
        if self.dim > (ns + nt):
            raise DimensionError('The maximum number of dimensions should be smaller than', (ns + nt))
        else:pass
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e * e.T
        # cal centering matrix H page 202 the last pargraph at left side
        n, _ = X.shape
        H = np.eye(n) - 1 / n * np.ones((n, n))
        # page 202 the last pargraph at right side
        matrix = (K @ L @ K + self.lamda * np.eye(n)) @ K @ H @ K.T
        # cal eigenvalues : w, eigenvectors :V
        w, V = scipy.linalg.eig(matrix)
        w, V = w.real, V.real
        # peak out the first self.dim components
        ind = np.argsort(abs(w))[::-1]
        A = V[:, ind[:self.dim]]
        # output the mapped data
        Z = K @ A
        Xs_new, Xt_new = Z[:ns, :], Z[ns:, :]
        return Xs_new, Xt_new


class DimensionError(Exception):
    pass