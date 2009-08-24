import numpy as np
import cov_prior
from mahalanobis_covariance import *
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step','MvNormalLR','lr_spatial','MVNLRMetropolis','lr_spatial_env']

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
    def p(val = bump_eigenvalues, vec = bump_eigenvectors, ctr = ctr, amp=amp):
        "A stupid hill, using Euclidean distance."
        return hill_fn(val, vec, ctr, amp)
        
    return locals()


# =======================================
# = The spatial-only, low-rank submodel =
# =======================================
def mod_matern(x,y,diff_degree,amp,scale,symm=False):
    """Matern with the mean integrated out."""
    return pm.gp.matern.geo_rad(x,y,diff_degree=diff_degree,amp=amp,scale=scale,symm=symm)+10000

def lrmvn_lp(x, m, piv, U, rl):
    """
    lrmvn_lp(x, mu, C)
    
    Low-rank multivariate normal log-likelihood
    """
    if rl != U.shape[0]:
        return -np.inf
    return pm.mv_normal_chol_like(x[piv[:rl]], m[piv[:rl]], U[:rl, :rl].T)
    
def rlrmvn(m, piv, U, rl):
    rl = U.shape[0]
    U_forrand = U[:,np.argsort(piv)]
    return np.dot(U_forrand.T, np.random.normal(size=rl))
    
MvNormalLR = pm.stochastic_from_dist('MvNormalLR', lrmvn_lp, rlrmvn, mv=True)

class MVNLRMetropolis(pm.AdaptiveMetropolis):
    def __init__(self, mvnlr, cov=None, delay=1000, scales=None, interval=200, greedy=True, shrink_if_necessary=False, verbose=0, tally=False):
        pm.AdaptiveMetropolis.__init__(self, mvnlr, cov, delay, scales, interval, greedy, shrink_if_necessary,verbose, tally)
        self.mvnlr = mvnlr
        self.piv = mvnlr.parents['piv']
        self.U = mvnlr.parents['U']
        self.rl = mvnlr.parents['rl']
    
    @classmethod
    def competence(cls,stochastic):
        if isinstance(stochastic, MvNormalLR):
            return 3
        else:
            return 0

    def covariance_adjustment(self, f=.9):
        """Multiply self.proposal_sd by a factor f. This is useful when the current proposal_sd is too large and all jumps are rejected.
        """
        pass

    def updateproposal_sd(self):
        """Compute the Cholesky decomposition of self.C."""
        pass
            
    def propose(self):
        
        # from IPython.Debugger import Pdb
        # Pdb(color_scheme='Linux').set_trace()   
        
        piv = self.piv.value
        v = self.mvnlr.value[piv[:self.rl]]
        jump = pm.rmv_normal_cov(np.zeros(self.rl), self.C[piv[:self.rl], :][:,piv[:self.rl]])
        if self.verbose > 2:
            print 'Jump :', jump

        new_val = np.empty(self.mvnlr.value.shape)
        
        # Jump the full-rank part
        new_val_fr = v + jump
        new_val[piv[:self.rl]] = new_val_fr
        
        # Jump the determined part
        ind_norms = pm.gp.trisolve(self.U.value[:, :self.rl], new_val_fr, uplo='U', transa='T')
        new_val[piv[self.rl:]] = np.dot(self.U.value[:, self.rl:].T, ind_norms) 

        self.mvnlr.value = new_val
        
class LRP(object):
    """A closure that can evaluate a low-rank field."""
    def __init__(self, x_fr, C, krige_wt):
        self.x_fr = x_fr
        self.C = C
        self.krige_wt = krige_wt
    def __call__(self, x):
        f_out = np.dot(np.asarray(self.C(x,self.x_fr)), self.krige_wt)
        return pm.invlogit(f_out).reshape(x.shape[:-1])
        
def lr_spatial(rl=50,**stuff):
    """A low-rank spatial-only model."""
    amp = pm.Exponential('amp',.1,value=1)
    scale = pm.Exponential('scale',.1,value=1.)
    diff_degree = pm.Uniform('diff_degree',0,2,value=.5)

    pts_in = stuff['pts_in']
    pts_out = stuff['pts_out']
    x_eo = np.vstack((pts_in, pts_out))

    @pm.deterministic
    def C(amp=amp,scale=scale,diff_degree=diff_degree):
        return pm.gp.Covariance(mod_matern, amp=amp, scale=scale, diff_degree=diff_degree)

    @pm.deterministic(trace=False)
    def ichol(C=C, rl=rl, x=x_eo):
        return C.cholesky(x, rank_limit=rl, apply_pivot=False)

    piv = pm.Lambda('piv', lambda d=ichol: d['pivots'])
    U = pm.Lambda('U', lambda d=ichol: d['U'].view(np.ndarray), trace=False)

    # Trace the full-rank locations
    x_fr = pm.Lambda('x_fr', lambda d=ichol, rl=rl, x=x_eo: x[d['pivots'][:rl]])

    # Evaluation of field at expert-opinion points
    f_eo = MvNormalLR('f_eo', np.zeros(x_eo.shape[0]), piv, U, rl, value=np.zeros(x_eo.shape[0]), trace=False)

    in_prob = pm.Lambda('in_prob', lambda f_eo=f_eo, n_in=pts_in.shape[0]: np.mean(pm.invlogit(f_eo[:n_in])))
    out_prob = pm.Lambda('out_prob', lambda f_eo=f_eo, n_in=pts_in.shape[0]: np.mean(pm.invlogit(f_eo[n_in:])))    

    @pm.deterministic(trace=False)
    def krige_wt(f_eo = f_eo, piv=piv, U=U, rl=rl):
        U_fr = U[:rl,:rl]
        f_fr = f_eo[piv[:rl]]
        return pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    p = pm.Lambda('p', lambda x_fr=x_fr, C=C, krige_wt=krige_wt: LRP(x_fr, C, krige_wt))

    return locals()            
    
# ======================================
# = Spatial and environmental low-rank =
# ======================================
def mod_spatial_mahalanobis(x,y,amp,val,vec,symm=False):
    return spatial_mahalanobis_covariance(x,y,amp,val,vec,symm)+10000

def lr_spatial_env(rl=50,**stuff):
    """A low-rank spatial-only model."""
    amp = pm.Exponential('amp',.1,value=1)

    pts_in = np.hstack((stuff['pts_in'],stuff['env_in']))
    pts_out = np.hstack((stuff['pts_out'],stuff['env_out']))
    x_eo = np.vstack((pts_in, pts_out))
    
    n_env = stuff['env_in'].shape[1]
    
    val = pm.Gamma('val',4,4,value=np.ones(n_env+1))
    vec = cov_prior.OrthogonalBasis('vec',n_env+1,constrain=False)

    @pm.deterministic
    def C(amp=amp,val=val,vec=vec):
        return pm.gp.Covariance(mod_spatial_mahalanobis, amp=amp, val=val, vec=vec)

    C.value(x_eo,x_eo)

    @pm.deterministic(trace=False)
    def ichol(C=C, rl=rl, x=x_eo):
        return C.cholesky(x, rank_limit=rl, apply_pivot=False)

    piv = pm.Lambda('piv', lambda d=ichol: d['pivots'])
    U = pm.Lambda('U', lambda d=ichol: d['U'].view(np.ndarray), trace=False)

    # Trace the full-rank locations
    x_fr = pm.Lambda('x_fr', lambda d=ichol, rl=rl, x=x_eo: x[d['pivots'][:rl]])

    # Evaluation of field at expert-opinion points
    f_eo = MvNormalLR('f_eo', np.zeros(x_eo.shape[0]), piv, U, rl, value=np.zeros(x_eo.shape[0]), trace=False)

    in_prob = pm.Lambda('in_prob', lambda f_eo=f_eo, n_in=pts_in.shape[0]: np.mean(pm.invlogit(f_eo[:n_in])))
    out_prob = pm.Lambda('out_prob', lambda f_eo=f_eo, n_in=pts_in.shape[0]: np.mean(pm.invlogit(f_eo[n_in:])))    

    @pm.deterministic(trace=False)
    def krige_wt(f_eo = f_eo, piv=piv, U=U, rl=rl):
        U_fr = U[:rl,:rl]
        f_fr = f_eo[piv[:rl]]
        return pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    p = pm.Lambda('p', lambda x_fr=x_fr, C=C, krige_wt=krige_wt: LRP(x_fr, C, krige_wt))

    return locals()