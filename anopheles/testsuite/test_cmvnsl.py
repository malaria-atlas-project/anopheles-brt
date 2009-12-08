from numpy.testing import *
import nose,  warnings
from anopheles import CMVNLStepper
import numpy as np
import pymc as pm
import pylab as pl

n = 301
V = np.eye(n*3)+1
S = np.linalg.cholesky(V)
M = np.zeros(n*3)
f = pm.MvNormalChol('f',M,S)

new_val = f.value.copy()
new_val[n:2*n] = 1
f.value = new_val

B = np.hstack((np.zeros((n,n)), -np.eye(n), np.zeros((n,n))))
Bl = np.vstack((np.hstack((np.eye(n), np.zeros((n,n)), np.zeros((n,n)))),np.hstack((np.zeros((n,n)), np.zeros((n,n)), np.eye(n)))))

n_neg = np.repeat(1,2*n)

acc = []
rej = []
sm = CMVNLStepper(f, B, np.zeros(n), Bl, n_neg, .999, S, M, n_cycles=1000, pri_S_type='tri')
pl.clf()
for i in xrange(10):
    sm.step()
    acc.append(sm.accepted)
    rej.append(sm.rejected)
    pl.plot(f.value)

