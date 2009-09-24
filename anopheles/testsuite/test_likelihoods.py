from numpy.testing import *
import nose,  warnings
import numpy as np
import pymc as pm
from scipy import integrate
import anopheles

npix = 5
prob_detect = .1
n_obs = 8

def unequal_binomial_lp(p):
    "A pure-Python version of ubl"
    n=len(p)
    lp = np.log(p)
    lomp = np.log(1.-p)
    
    out = np.zeros(n+1)
    
    out[0] = lomp[0]
    out[1] = lp[0]
    for i in range(1,n):
        last = out.copy()
        out[i+1] = out[i] + lp[i]        
        for j in range(i,0,-1):
            if np.isinf(last[j-1]+lp[i]) and np.isinf(last[j]+lomp[i]):
                out[j]=-np.inf
            else:
                out[j] = pm.flib.logsum([last[j-1]+lp[i], last[j]+lomp[i]])
            if np.isnan(out[j]):
                raise ValueError
        out[0] += lomp[i]
            
    return out

def standard_things(q,prob_detect=prob_detect,n_obs=n_obs):
    
    npix=len(q)
    
    # Log-distribution of number of pixels positive.        
    lpf = anopheles.utils.ubl(q)
    lpp = unequal_binomial_lp(q)
    assert_equal(lpf, lpp)
    
    # Binomial mixture from Fortran.
    pbf = np.exp([anopheles.utils.bin_ubl(x,n_obs,prob_detect,q) for x in xrange(n_obs+1)])

    # Do binomial mixture by hand.
    pbp = np.zeros(n_obs+1)
    for i in xrange(npix+1):
        pbp+=np.exp([pm.binomial_like(x,n_obs,prob_detect*float(i)/npix)+lpf[i] for x in xrange(n_obs+1)])

    assert_almost_equal(pbf,pbp)
    
    return lpf, lpp, pbf, pbp

class test_multipoint_likelihoods(object):

    def test_multiple_obs(self):
        "Tests bin_ubls... should be a slam dunk"
        ps = np.array([.1, .8, .4, .7, .2, .13])
        q = .4
        npos = np.array([10, 13, 2])
        ns = np.array([100, 15, 3])
        breaks = np.array([0,1,3,6])
        
        meth1 = np.sum([anopheles.bin_ubl(npos[i], ns[i], q, ps[breaks[i]:breaks[i+1]]) for i in [0,1,2]])
        meth2 =anopheles.bin_ubls(npos, ns, q, breaks, ps)
        
        assert_equal(meth1,meth2)
        

    def test_random(self):
        "Does a bunch of trials to look for segfaults."
        for i in xrange(1000):
            prob_detect = np.random.random()
            npix = np.random.randint(1,10)
            n_obs = np.random.randint(1,10)
            q = np.random.random(size=npix)
                    
            lpf, lpp, pbf, pbp = standard_things(q,prob_detect,n_obs)
        
    def test_binomial_case(self):
        """Checks for correspondence with the binomial distribution in the case of equal
        presence probabilities."""

        q = np.ones(5)*.2
        lpf, lpp, pbf, pbp = standard_things(q)

        # In this case you can compute the log-p of number of pixels positive directly.
        lpo = np.array([pm.binomial_like(x,npix,q[0]) for x in range(npix+1)])    

        assert_almost_equal(lpf, lpo)
        assert_almost_equal(lpp, lpo)
    
    def test_withzeros(self):
        "Makes sure no NaN's happen when some probabilities are zero."
        q = np.zeros(npix)
        q[0]=.99

        # Log-distribution of number of pixels positive.        
        lpf = anopheles.utils.ubl(q)
        lpp = unequal_binomial_lp(q)
        pp = np.exp(lpp)
        pf = np.exp(lpf)
        assert_equal(pp,pf)

        lpb = anopheles.utils.bin_ubl(3,8,.1,q)

        assert(not np.isnan(lpb))
        assert(not np.any(np.isnan(pf)))
        
        # Binomial mixture from Fortran.
        pbf = np.exp([anopheles.utils.bin_ubl(x,n_obs,prob_detect,q) for x in xrange(n_obs+1)])
        
        # Do binomial mixture by hand.
        pbp = np.zeros(n_obs+1)
        for i in xrange(npix+1):
            pbp+=np.exp([pm.binomial_like(x,8,prob_detect*float(i)/npix)+lpf[i] for x in xrange(n_obs+1)])
            
        assert_almost_equal(pbf,pbp)
        
        
if __name__ == '__main__':
    nose.runmodule()

