from anopheles import spatial_mahalanobis_covariance
import numpy as np
import pymc as pm

N = 5
n_env = 2

lon = np.random.uniform(-np.pi,np.pi,size=N)
lat = np.random.uniform(0,np.pi/2.,size=N)
env = []
for i in xrange(n_env):
    env.append(np.random.normal(size=N))
    
x = np.vstack([lon,lat]+env).T

val = np.ones(n_env+1)
vec = np.eye(n_env+1)

C = spatial_mahalanobis_covariance(x,x,1,val,vec,symm=True)

import pylab as pl
pl.clf()
pl.imshow(C.view(np.ndarray),interpolation='nearest')