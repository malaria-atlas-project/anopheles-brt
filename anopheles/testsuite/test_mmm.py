from anopheles import mod_matern_with_mahal
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

C = mod_matern_with_mahal(x,x,)