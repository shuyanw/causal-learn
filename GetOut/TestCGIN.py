import sys

sys.path.append("")
import unittest

import numpy as np
import matplotlib.pyplot as plt
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma


class TestGIN(unittest.TestCase):

    def test_case1(self):
        sample_size = 1000
        np.random.seed(0)

        L1 = np.random.uniform(-2, 2, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * np.sin(L1) + np.random.uniform(-1, 1, size=sample_size) ** 5

        X2 = np.random.uniform(0.5, 2.0) * np.sin(L1) + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.sin(L1) + np.random.uniform(-1, 1, size=sample_size) ** 5

        X4 = np.random.uniform(0.5, 2.0) * np.sin(L1) + np.random.uniform(-1, 1, size=sample_size) ** 5
        plt.plot(L1,X1,'go',L1,X2, 'ro', L1, X3,'bo',L1,X4, 'yo',markersize = 0.8)
        plt.show()
        X5 = np.random.uniform(0.5, 2.0) * np.sqrt(abs(L2)) + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * np.sqrt(abs(L2)) + np.random.uniform(-1, 1, size=sample_size) ** 5
        X7 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        plt.plot(L2,X5,'go',L2,X6, 'ro', L2, X7,'bo',L2,X8, 'yo',markersize = 0.8)
        plt.show()

        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8]).T
        #print(data.shape)
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)
'''
    def test_case1(self):
        sample_size = 1000
        np.random.seed(0)
        L3 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L1 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5

        data = np.array([X1, X2, X3, X4]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        #print(g, k, hsic_test_gamma(L1, L3)[0],hsic_test_gamma(X1, X2)[0],  hsic_test_gamma(X1, X3)[0], hsic_test_gamma(X1, X6)[0],)
    
    def test_case2(self):
        print('run test2')
        sample_size = 1000
        np.random.seed(0)
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1,
                                                                                                     size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X7 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X9 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5

        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)
    '''

'''    
    def test_case3(self):
        #print('run test3')
        sample_size = 2000
        np.random.seed(0)
        #cycles between blocks
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L4 = np.random.uniform(-1, 1, size=sample_size) ** 5
        L2 = np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L3 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        
        
        X1 =  L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = 0.5 * L1 + np.random.uniform(0.5, 2.0) * L2 +  np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = 0.25 * L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = L1 + np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X7 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        Y1 = X1- 2*X2
        Y2 = X4 -4*X3
        data = np.array([Y1, Y2, X5, X6]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)

        


        

        L5 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1,size=sample_size) ** 5


        
        X9 = np.random.uniform(0.5, 2.0) * L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X10 = np.random.uniform(0.5, 2.0) *  L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1,  size=sample_size) ** 5
 
        #e_xz = np.dot(omega, data[:, X].T)


        #X3 = X3-2*X4
        #X5 = X5-X6
        data = np.array([X1, X2, X3, X4,]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        cov = np.cov(data.T)
        #print(hsic_test_gamma(L1, L3)[0], hsic_test_gamma(X5, X3)[0], hsic_test_gamma(X1, X3 - 2 * X4)[0], )
        cov_m = cov[np.ix_(Z, X)]
        _, _, v = np.linalg.svd(cov_m)
        omega = v.T[:, -1]
        print(omega)
        #g, k = GIN(data)
        #print(g, k)
'''

        

        



