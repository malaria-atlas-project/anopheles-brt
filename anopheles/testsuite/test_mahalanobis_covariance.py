import anopheles
import pymc as pm
import numpy as np
import pylab as pl

mc = anopheles.mahalanobis_covariance

n=101
x = np.atleast_2d([0,0])
y = np.dstack(np.meshgrid(np.linspace(-2,2,n),np.linspace(-2,2,n)))

dd = 1.5
amp = 1
val = np.ones(2)
vec = np.eye(2)
C = pm.gp.FullRankCovariance(mc, diff_degree=dd, amp=amp, val=val, vec=vec)

Csurf = np.asarray(C(x,y)).reshape((n,n))
pl.clf()
pl.imshow(Csurf,interpolation='nearest')
pl.colorbar()

# mc(x,y,diff_degree,amp,val,vec,symm=None)