# Copyright (C) 2009  Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pymc as pm
from utils import mahal
from scipy.special import gamma

__all__ = ['mahalanobis_covariance']


def mahalanobis_covariance(x,y,diff_degree,amp,val,vec,symm=None):
    """
    Converts x and y to a matrix of covariances. x and y are assumed to have
    columns (long,lat,t). Parameters are:
    - t_gam_fun: A function returning a matrix of variogram values.
      Inputs will be the 't' columns of x and y, as well as kwds.
    - amp: The MS amplitude of realizations.
    - scale: Scales distance.
    - inc, ecc: Anisotropy parameters.
    - n_threads: Maximum number of threads available to function.
    - symm: Flag indicating whether matrix will be symmetric (optional).
    - kwds: Passed to t_gam_fun.
    
    Output value should never drop below -1. This happens when:
    -1 > -sf*c+k
    
    """
    # Allocate 
    nx = x.shape[0]
    ny = y.shape[0]
    ndim = x.shape[1]
    
    C = np.asmatrix(np.empty((nx,ny),order='F'))
    
    # Figure out symmetry and threading
    if symm is None:
        symm = (x is y)

    n_threads = min(pm.get_threadpool_size(), nx*ny / 2000)        
    
    if n_threads > 1:
        if not symm:
            bounds = np.linspace(0,ny,n_threads+1)
        else:
            bounds = np.array(np.sqrt(np.linspace(0,ny*ny,n_threads+1)),dtype=int)

    # Target function for threads
    def targ(C,x,y,symm,diff_degree,amp,val,vec,cmin,cmax):
        mahal(C,x,y,symm,diff_degree,gamma(diff_degree),amp,val,vec,cmin,cmax)
    
    # Dispatch threads        
    if n_threads <= 1:
        mahal(C,x,y,symm,diff_degree,gamma(diff_degree),amp,val,vec,cmin=0,cmax=C.shape[1])
    else:
        thread_args = [(C,x,y,symm,diff_degree,amp,val,vec,bounds[i],bounds[i+1]) for i in xrange(n_threads)]
        pm.map_noreturn(targ, thread_args)

    if symm:
        pm.gp.symmetrize(C)

    return C
    
if __name__ == '__main__':
    nd = 20
    x = np.linspace(-1,1,1001)
    xo = np.vstack((x,)*nd).T
    # x,y = np.meshgrid(x,x)
    # z = np.empty((x.shape[0],x.shape[0],2))
    # z[:,:,0] = x
    # z[:,:,1] = y
    
    C = pm.gp.Covariance(mahalanobis_covariance, amp=1, val=np.ones(nd), vec=np.array(np.eye(nd),order='F')*2.)
    C(xo,xo)
    # A = np.asarray(C(np.atleast_2d(np.zeros(nd)),z)).reshape((x.shape[0],x.shape[0]))
    from pylab import *
    close('all')
    # imshow(A)
    # title('Autocov')
    # figure()
    # imshow(C(xo,xo).view(np.ndarray))
    s=C.cholesky(xo)
    # title('Evaluation')