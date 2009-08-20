from numpy.testing import *
import nose,  warnings
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

class test_spatial_mahalanobis_covariance(object):
    def test_spatial_only(self):

        val = np.ones(n_env+1)
        vec = np.eye(n_env+1)
        val[1:]=1e10

        C1 = spatial_mahalanobis_covariance(x,x,1,val,vec,symm=True)
        C2 = pm.gp.exponential.geo_rad(x[:,:2],x[:,:2],amp=1,scale=1,symm=True)
    
        assert_almost_equal(C1,C2)
    
    def test_env_only(self):
        val = np.ones(n_env+1)
        vec = np.eye(n_env+1)
        val[0] = 1e10

        C1 = spatial_mahalanobis_covariance(x,x,1,val,vec,symm=True)
        C2 = pm.gp.exponential.euclidean(x[:,2:],x[:,2:],amp=1,scale=1,symm=True)
    
        assert_almost_equal(C1,C2)
    
    def test_anisotropic_env_only(self):
        B = np.random.normal(size=(n_env,n_env))
        e_val,e_vec = np.linalg.eigh(np.dot(B,B.T))
    
        val = np.empty(n_env+1)
        val[0]=1e10
        val[1:]=e_val

        vec = np.zeros((n_env+1,n_env+1))
        vec[1:,1:]=e_vec
        vec[0,0]=1
    
        C1 = spatial_mahalanobis_covariance(x,x,1,val,vec,symm=True)
        x_trans = np.dot(x[:,2:],e_vec)/np.sqrt(e_val)
        C2 = pm.gp.exponential.euclidean(x_trans,x_trans,amp=1,scale=1,symm=True)
    
        assert_almost_equal(C1,C2)

if __name__ == '__main__':
    nose.runmodule()
