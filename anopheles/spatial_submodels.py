import numpy as np
import cov_prior
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step']

def hinge(x, cp):
    "A MaxEnt hinge feature"
    y = x-cp
    y[x<cp]=0.
    return y
    
def step(x,cp):
    "A MaxEnt step feature"
    y = np.ones(x.shape)
    y[x<cp]=0.
    return y

# TODO: 'Inducing points' should be highest-rank points from random samples from 
# the EO and not-EO regions, so that the data locations don't influence the prior.
# The rest of the field should be determined by the EO. Don't use the extra nugget.

# def make_model(session, species, covariates, rl=500):
#     sites, eo = species_query(session, species)
#     # TODO: Space-only model with forcible rank depression. Can put in covariates later.
#     # Remember to use expert opinion and background.
#     # TODO: Probably don't use improved approximations of Quinonero-Candela and Rasmussen.
#     # Reason: no sense adding an extra nugget.
#     
#     p_find = pm.Uninformative('p_find',0,1)
#     
#     mahal_eigenvalues = pm.Gamma('mahal_eigenvalues', 2, 2, size=len(covariates))
#     mahal_eigenvectors = cov_prior.OrthogonalBasis('mahal', len(covariates))
#     
#     # The Mahalanobis covariance
#     @deterministic
#     def C(val=mahal_eigenvalues, vec=mahal_eigenvectors, amp=amp):
#         return pm.gp.Covariance(mahalanobis_covariance,amp,val,vec)
#         
#     # The low-rank Cholesky decomposition of the Mahalanobis covariance
#     @deterministic(trace=False)
#     def S(C=C,xtot=xtot,rl=rl):
#         return C.cholesky(xtot,rank_limit=rl)


class hill_fn(object):
    "Closure used by ghetto_spatial_submodel"    
    def __init__(self, val, vec, ctr, amp):
        self.val = val
        self.vec = vec
        self.ctr = ctr
        self.amp = amp
    def __call__(self, x):
        dev = x-self.ctr
        tdev = np.dot(dev, self.vec)
        if len(dev.shape)==1:
            ax=0
        else:
            ax=1
        return pm.invlogit(np.sum(tdev**2/self.val,axis=ax)*self.amp)


def spatial_hill(**kerap):
    "For debugging only"
    amp = pm.Uninformative('amp',-1)
    
    @pm.stochastic
    def ctr(value=np.array([0,0])):
        "This makes the center uniformly distributed over the surface of the earth."
        if value[0] < -np.pi or value[0] > np.pi or value[1] < -np.pi/2. or value[1] > np.pi/2.:
            return -np.inf
        return np.cos(value[1])

    bump_eigenvalues = pm.Gamma('bump_eigenvalues', 2, 2, size=2)
    bump_eigenvectors = cov_prior.OrthogonalBasis('bump_eigenvectors',2)
    
    @pm.deterministic
    def f(val = bump_eigenvalues, vec = bump_eigenvectors, ctr = ctr, amp=amp):
        "A stupid hill, using Euclidean distance."
        return hill_fn(val, vec, ctr, amp)
        
    return locals()