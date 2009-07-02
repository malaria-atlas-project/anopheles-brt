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


# =============================
# = The standard spatial hill =
# =============================
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


# ==============================
# = A spatial-only low-rank GP =
# ==============================
class krige_fn(object):
    """docstring for krige_fn"""
    def __init__(self, val, x_fr, U):
        self.x_fr = x_fr
        self.U = U
        self.val = val
        self.shift(x_fr, U)        
        

    def shift(self, x_fr, U):
        "Shifts to a new x_fr or U."
        self.U = U
        self.x_fr = x_fr
        self.krige_wt = pm.gp.trisolve(self.U,pm.gp.trisolve(self.U,self.val,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    def __call__(self, x, U=None, x_fr=None):
        f_out = None

        # Possibly shift
        if U is not None:
            if U is not self.U:
                self.val = self(x_fr)                
                f_out = self.shift(x_fr, U)

        if f_out is None:
            f_out = np.asarray(np.dot(self.krige_wt,self.C(self.x_fr,x)))
        return pm.invlogit(x_out.ravel()).reshape(x.shape[:-1])


# FIXME: Fill these in!
class krige_fn_stepper(pm.AdaptiveMetropolis):
    pass
    
class KrigeFn(pm.Stochastic):
    def __init__(self, name, x_fr, U):
        self.rl = x_fr.shape[0]
        def lpf(value, x, U):
            return pm.mv_normal_chol_like(value(x, U), np.zeros(self.rl), U)
        pm.Stochastic.__init__(self, name, {'x': x_fr, 'U': U}, logp=lpf)
        

def mod_matern(x,y,diff_degree,amp,scale,symm=False):
    "Matern with the mean integrated out."
    return pm.gp.matern.geo_rad(x,y,diff_degree=diff_degree,amp=amp,scale=scale,symm=symm)+10000
    
def lr_spatial(rl=50,**stuff):
    "A low-rank spatial-only model."
    amp = pm.Exponential('amp',.1,value=1)
    scale = pm.Exponential('scale',.1,value=1.)
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

    f = KrigeFn('f', x_fr, U)
    
    
    return locals()            
    
    
    