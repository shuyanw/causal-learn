import sys

sys.path.append("")
import unittest

import numpy as np

from causallearn.search.HiddenCausal.CGIN.GIN import GIN
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma


class TestGIN(unittest.TestCase):
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
        print(g, k, hsic_test_gamma(L1, L3)[0],hsic_test_gamma(X1, X2)[0],  hsic_test_gamma(X1, X3)[0], hsic_test_gamma(X1, X6)[0],)

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
    
    def test_case3(self):
        print('run test3')
        sample_size = 2000
        np.random.seed(0)
        L1 = np.random.uniform(-1, 1, size=sample_size) ** 5
        X1 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(-1, 1, size=sample_size) ** 5

        L4 = np.random.uniform(-1, 1, size=sample_size) ** 5

        L2 = np.random.uniform(0.5, 2.0) * L1 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        L3 = np.random.uniform(0.5, 2.0) * L2 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X3 = L2 + np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X4 = 0.5 * L2 + np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X5 = 2.0 * L2 + np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X6 = 2.0 * L2 + np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1, size=sample_size) ** 5



        L5 = np.random.uniform(0.5, 2.0) * L3 + np.random.uniform(-1, 1,size=sample_size) ** 5


        X7 = np.random.uniform(0.5, 2.0) * L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X8 = np.random.uniform(0.5, 2.0) * L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X9 = np.random.uniform(0.5, 2.0) * L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1, size=sample_size) ** 5
        X10 = np.random.uniform(0.5, 2.0) *  L5 + np.random.uniform(0.5, 2.0) * L4 + np.random.uniform(-1, 1,  size=sample_size) ** 5


        X3 = X3-2*X4
        X5 = X5-X6

        #print(hsic_test_gamma(L1, L3)[0], hsic_test_gamma(X5, X3)[0], hsic_test_gamma(X1, X3 - 2 * X4)[0], )
        data = np.array([X3, X5, X7, X9]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)

        '''
        data = np.array([X1, X2, X3, X4, X5, X6, X7, X8, X9]).T
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        g, k = GIN(data)
        print(g, k)
        '''



