import numpy as np
import cov_prior
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step','krige_fn','lr_spatial']

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
    "Closure used by spatial_hill"    
    def __init__(self, val, vec, ctr, amp):
        self.val = val
        self.vec = vec
        self.ctr = ctr
        self.amp = amp
        self.max_argsize = np.inf
        
    def __call__(self, x):
        xr = x.reshape(-1,2)
        dev = xr-self.ctr
        tdev = np.dot(dev, self.vec)
        if len(dev.shape)==1:
            ax=0
        else:
            ax=-1
        return pm.invlogit(np.sum(tdev**2/self.val,axis=ax)*self.amp).reshape(x.shape[:-1])


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

# FIXME: Fill these in!
class krige_fn(object):
    """docstring for krige_fn"""
    def __init__(self, krige_wt, x_fr, C):
        self.krige_wt = krige_wt
        self.x_fr = x_fr
        self.C = C
    def __call__(self, x):
        return pm.invlogit(np.asarray(np.dot(self.krige_wt,self.C(self.x_fr,x))).ravel()).reshape(x.shape[:-1])

def mod_matern(x,y,diff_degree,amp,scale):
    "Matern with the mean integrated out."
    return pm.gp.matern.geo_rad(x,y,diff_degree=diff_degree,amp=amp,scale=scale)+10000
    
def lr_spatial(rl=50,**stuff):
    "A low-rank spatial-only model."
    amp = pm.Exponential('amp',.1,value=1)
    scale = pm.Exponential('scale',.1,value=.01)
    diff_degree = pm.Uniform('diff_degree',0,2,value=.5)
    
    pts_in = stuff['pts_in']
    pts_out = stuff['pts_out']
    x = np.vstack((pts_in, pts_out))
    
    @pm.deterministic
    def C(amp=amp,scale=scale,diff_degree=diff_degree):
        return pm.gp.Covariance(mod_matern, amp=amp, scale=scale, diff_degree=diff_degree)
        
    @pm.deterministic(trace=False)
    def x_and_U(C=C, rl=rl, x=x):
        d = C.cholesky(x, rank_limit=rl, apply_pivot=False)
        piv = d['pivots']
        U = d['U']
        return x[piv[:U.shape[0]]], U 
    
    # Trace the full-rank locations
    x_fr = pm.Lambda('x_fr', lambda t=x_and_U: t[0])
    # Don't trace the Cholesky factor. It may be big.
    U = x_and_U[1]
    
    @pm.potential
    def fr_check(U=U, rl=rl):
        if U.shape[0]==rl:
            return 0.
        else:
            return -np.inf
    
    f_mesh = pm.Uninformative('f_mesh',np.zeros(rl))
    
    @pm.deterministic
    def krige_wt(x_fr=x_fr, U=U, rl=rl, C=C, f_mesh=f_mesh):
        return pm.gp.trisolve(U,pm.gp.trisolve(U,f_mesh,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    @pm.potential
    def f_logp(U=U, krige_wt=krige_wt, f_mesh=f_mesh, rl=rl):
        return -.5*np.sum(np.log(2.*np.pi) + np.log(np.diag(U))) - .5*np.dot(f_mesh, krige_wt)
    
    # The 'rest' of the GP
    @pm.deterministic
    def f(krige_wt=krige_wt, x_fr=x_fr, C=C):
        return krige_fn(krige_wt, x_fr, C)
        
    return locals()
        
            
    
    
    