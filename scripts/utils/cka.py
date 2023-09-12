"""Centered Kernel/Linear Alignment

https://github.com/jayroxis/CKA-similarity/blob/main/CKA.ipynb

np_cka = CKA()

X = np.random.randn(10000, 100)
Y = np.random.randn(10000, 100)

print('Linear CKA, between X and Y: {}'.format(np_cka.linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(np_cka.linear_CKA(X, X)))

print('RBF Kernel CKA, between X and Y: {}'.format(np_cka.kernel_CKA(X, Y)))
print('RBF Kernel CKA, between X and X: {}'.format(np_cka.kernel_CKA(X, X)))
"""
import math

import numpy as np


class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        Ii = np.eye(n)
        H = Ii - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)
