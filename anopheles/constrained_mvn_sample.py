import numpy as np
import pymc as pm
from utils import lcg, lcm

__all__ = ['cmvns','cmvns_l']

def cmvns_l(cur_val, B, y, Bl, n_neg, p_find, pri_S = None, pri_M = None, n_cycles=1, pri_S_type='square'):
    """
    Metropolis samples cur_val, under the constraint that B*cur_val < y, 
    with likelihood term corresponding to n_neg negative observations independent
    with probabilities p_find if Bl*cur_val>0, else 0.
    """
    
    cur_val = np.asarray(cur_val).squeeze()
    B = np.asarray(B)
    Bl = np.asarray(Bl)
    
    # Change coordinates so that the elements of cur_val are standard normal.
    if pri_M is not None:
        pri_M =np.asarray(pri_M).squeeze()
        cur_val = cur_val - pri_M
    
    if pri_S is not None:
        pri_S = np.asarray(pri_S).squeeze()
        if pri_S_type == 'square':
            new_val = np.linalg.solve(pri_S, cur_val)
            B = np.dot(B,pri_S)
            Bl = np.dot(Bl, pri_S)
        elif pri_S_type == 'diag':
            new_val = cur_val / pri_S
            B = B*pri_S
            Bl = Bl*pri_S
        elif pri_S_type == 'tri':
            new_val = pm.gp.trisolve(pri_S, cur_val, uplo='L', transa='N')
            B = np.dot(B,pri_S)            
            Bl = np.dot(Bl,pri_S)
        else:
            raise ValueError, 'Prior matrix square root type %s not recognized.'%pri_S_type
    else:
        new_val = cur_val.copy()
    
    if np.any(np.dot(B,new_val) > y):
        # Adjust in case of numerical problems.
        new_val = new_val + 1e-5
        if np.any(np.dot(B,new_val) > y):
            raise ValueError, 'Starting values do not satisfy constraints.'
    
    # Do the specified number of cycles.
    n = len(cur_val)
    y_ = y-np.dot(B,new_val)
    lop = np.dot(Bl,new_val)
    u=np.random.random(size=(n,n_cycles)).copy('F')
    um=np.random.random(size=(n,n_cycles)).copy('F')
    # Call to Fortran routine lcg, which overwrites new_val in-place. Number of
    # cycles is determined by size of u.
    acc, rej = lcm(np.asarray(B,order='F'), y_, new_val, u, np.asarray(Bl,order='F'), n_neg, p_find, um, lop)
    
    # Change back to original coordinates and return.
    if pri_S is not None:
        if pri_S_type == 'square' or pri_S_type == 'tri':
            new_val = np.dot(pri_S, new_val)
        else:
            new_val *= pri_S    
    
    return new_val, acc, rej
    
def cmvns(cur_val, B, y, pri_S=None, pri_M=None, n_cycles=1, pri_S_type='square'):
    """
    Gibbs samples cur_val, under the constraint that B*cur_val < y.
    Makes use of pri_S, a matrix square root of the prior covariance; and
    pri_M, the prior mean.
    """
    
    # Change coordinates so that the elements of cur_val are standard normal.
    if pri_M is not None:
        cur_val = cur_val - pri_M
    
    if pri_S is not None:
        if pri_S_type == 'square':
            new_val = np.linalg.solve(pri_S, cur_val)
            B = np.dot(B,pri_S)
        elif pri_S_type == 'diag':
            new_val = cur_val / pri_S
            B = B*pri_S
        elif pri_S_type == 'tri':
            B = np.dot(B,pri_S)            
            raise NotImplementedError
        else:
            raise ValueError, 'Prior matrix square root type %s not recognized.'%pri_S_type
    else:
        new_val = cur_val.copy()
    
    if np.any(np.dot(B,new_val) > y):
        raise ValueError, 'Starting values do not satisfy constraints.'
    
    # Do the specified number of cycles.
    n = len(cur_val)
    y_ = y-np.dot(B,new_val)
    u=np.random.random(size=(n,n_cycles)).copy('F')
    # Call to Fortran routine lcg, which overwrites new_val in-place. Number of
    # cycles is determined by size of u.
    lcg(np.asarray(B,order='F'), y_, new_val, u)        
    
    # Change back to original coordinates and return.
    if pri_S is not None:
        if pri_S_type == 'square' or pri_S_type == 'tri':
            new_val = np.dot(pri_S, new_val)
        else:
            new_val *= pri_S    
    
    return new_val
    
    
if __name__ == '__main__':
    theta = np.pi*2/3.*0
    B = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    y = np.array([9,9])*.333
    n = 1000

    n_like = 10
    n_neg = np.ones(n_like)
    # B_like = np.random.normal(size=(n_like, 2))
    B_like = np.ones((n_like,2))
    # B_like[:,1]=0
    p_find = .8

    vals = np.zeros((n,2))*-10
    for i in xrange(n):
        # vals[i,:] = cmvns(vals[i,:], B, y, n_cycles=100)
        vals[i,:] = cmvns_l(vals[i,:], B, y, B_like, n_neg, p_find, n_cycles=100)
        
    import pylab as pl
    pl.close('all')
    vals = np.dot(B,vals.T).T
    
    likelops = np.dot(B_like, vals.T).T
    likeps = pm.invlogit(likelops)
    # print np.mean(likeps, axis=0)
    # pl.hist(,50)
    pl.plot(likeps)
    pl.figure()
    pl.plot(vals[:,0],vals[:,1],'k.')