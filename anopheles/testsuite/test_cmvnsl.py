from numpy.testing import *
import nose,  warnings
from anopheles import spatial_mahalanobis_covariance
import numpy as np
import pymc as pm

n = 31
V = np.eye(n*3)+1
S = np.linalg.cholesky(V)
f = pm.MvNormalChol('f',np.zeros(n*3),S)

B = np.hstack((np.zeros(n), np.eye(n), np.zeros(n)))
Bl = np.vstack((np.hstack((np.eye(n), np.zeros(n), zeros(n))),np.hstack((np.zeros(n), np.zeros(n), np.eye(n)))))

n_neg = 0

# sm = CMVNLStepper(f, B, y, Bl, n_neg, p_find, pri_S, pri_M, n_cycles=1, pri_S_type='square')

