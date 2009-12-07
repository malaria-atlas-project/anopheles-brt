import numpy as np
import cov_prior
from mahalanobis_covariance import *
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step','lr_spatial','lr_spatial_env','MVNLRParentMetropolis','minimal_jumps','bookend','spatial_env','nogp_spatial_env','normalize_env']

# =======================================
# = The spatial-only, low-rank submodel =
# =======================================
def mod_matern(x,y,diff_degree,amp,scale,symm=False):
    """Matern with the mean integrated out."""
    return pm.gp.matern.geo_rad(x,y,diff_degree=diff_degree,amp=amp,scale=scale,symm=symm)+10000
                
class LRP(object):
    """A closure that can evaluate a low-rank field."""
    def __init__(self, x_fr, C, krige_wt, f2p):
        self.x_fr = x_fr
        self.C = C
        self.krige_wt = krige_wt
        self.f2p = f2p
    def __call__(self, x, f2p=None, offdiag=None):
        if f2p is None:
            f2p = self.f2p
        if offdiag is None:
            offdiag = self.C(x,self.x_fr)
        return f2p(np.dot(np.asarray(offdiag), self.krige_wt).reshape(x.shape[:-1]))
        
def mod_spatial_mahalanobis(x,y,val,vec,const_frac,symm=False):
    return spatial_mahalanobis_covariance(x,y,1,val,vec,const_frac,symm)

def normalize_env(x, means, stds):
    x_norm = x.copy().reshape(-1,x.shape[-1])
    for i in xrange(2,x_norm.shape[1]):
        x_norm[:,i] -= means[i-2]
        x_norm[:,i] /= stds[i-2]
    return x_norm

class LRP_norm(LRP):
    """
    A closure that can evaluate a low-rank field.
    
    Normalizes the third argument onward.
    """
    def __init__(self, x_fr, C, krige_wt, means, stds, f2p):
        LRP.__init__(self, x_fr, C, krige_wt, f2p)
        self.means = means
        self.stds = stds

    def __call__(self, x,f2p=None,offdiag=None):
        x_norm = normalize_env(x, self.means, self.stds)
        return LRP.__call__(self, x_norm.reshape(x.shape), f2p,offdiag)


class fullcond_fr_sampler(object):
    def __init__(self, x_fr, x_eo, n_in, n_out, C, vals_in, vals_out, nugget):
        self.x_fr = x_fr
        self.x_eo = x_eo
        self.n_in = n_in
        self.n_out = n_out
        self.C = C
        self.nugget = nugget
        
        # from IPython.Debugger import Pdb
        # Pdb(color_scheme='Linux').set_trace()   
        self.U_eo = self.C.cholesky(self.x_eo, nugget=self.nugget*np.ones(self.n_in+self.n_out))
        offdiag = self.C(self.x_eo, self.x_fr)
        self.o_U_eo = pm.gp.trisolve(self.U_eo,offdiag,uplo='U',transa='T')
        C_cond = self.C(self.x_fr, self.x_fr)-np.dot(self.o_U_eo.T,self.o_U_eo)
        self.L_cond = np.linalg.cholesky(C_cond)
        self.obs_val = np.concatenate((vals_in * np.ones(self.n_in), vals_out*np.ones(self.n_out)))
        self.M_cond = np.dot(self.o_U_eo.T, pm.gp.trisolve(self.U_eo, self.obs_val, transa='T', uplo='U'))
        
    def __call__(self):
        return np.asarray(self.M_cond + np.dot(self.L_cond, np.random.normal(size=self.x_fr.shape[0]))).ravel()
        

def lr_spatial_env(rl=200,**stuff):
    """A low-rank spatial-only model."""

    n_env = stuff['env_in'].shape[1]
    x_fr = normalize_env(stuff['full_x_fr'], stuff['env_means'], stuff['env_stds'])
    pts_in = stuff['pts_in']
    f2p = stuff['f2p']
    
    valpow = pm.Uniform('valpow',0,10,value=.01, observed=False)
    valmean = pm.Lambda('valmean',lambda valpow=valpow : np.arange(n_env+1)*valpow)
    valV = pm.Exponential('valV',1,value=.1)

    # val = pm.Normal('val',valmean,1./valV,value=np.concatenate(([-2],-1*np.ones(n_env))))
    vals = [pm.Normal('val_%i'%i,valmean[i],1./valV,value=1) for i in xrange(n_env+1)]
    vals[0].value = 0
    val = pm.Lambda('val',lambda vals=vals: np.array(vals))
    baseval = pm.Exponential('baseval', .01, value=np.exp(-2), observed=False)
    expval = pm.Lambda('expval',lambda val=val,baseval=baseval:np.exp(val)*baseval)    

    vec = cov_prior.OrthogonalBasis('vec',n_env+1,constrain=True)

    const_frac = pm.Uniform('const_frac',0,1,value=.1)
    
    @pm.deterministic
    def C(val=expval,vec=vec,const_frac=const_frac):
        return pm.gp.FullRankCovariance(mod_spatial_mahalanobis, val=val, vec=vec, const_frac=const_frac)
    
    # TODO tomorrow: Gibbs sample through to initialize to a good, constraint-satisfying state.   
    # Forget that, just Gibbs sample with constraint satisfaction! 
    @pm.deterministic
    def fullcond_sampler(C=C, vals_in=.25, vals_out=-.25, nugget=.01):
        try:
            return fullcond_fr_sampler(x_fr, normalize_env(stuff['full_x_eo'], stuff['env_means'], stuff['env_stds']),stuff['n_in'],stuff['n_out'],C,vals_in,vals_out,nugget)
        except np.linalg.LinAlgError:
            return None
    

    @pm.deterministic(trace=False)
    def U_fr(C=C, x=x_fr):
        try:
            return C.cholesky(x)
        except np.linalg.LinAlgError:
            return None

    @pm.potential
    def rank_check(U=U_fr):
        if U is None:
            return -np.inf
        else:
            return 0.

    L_fr = pm.Lambda('L',lambda U=U_fr: U.T, trace=False)

    # Evaluation of field at expert-opinion points
    init_val = np.ones(len(x_fr))*-.1
    # init_val[len(init_val)/2]=-1
    f_fr = pm.MvNormalChol('f_fr', np.zeros(len(x_fr)), L_fr, value=init_val)

    @pm.deterministic(trace=False)
    def krige_wt(f_fr=f_fr, U_fr=U_fr):
        return pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    p = pm.Lambda('p', lambda x_fr=x_fr, C=C, krige_wt=krige_wt, means=stuff['env_means'], stds=stuff['env_stds'], f2p=f2p: LRP_norm(x_fr, C, krige_wt, means, stds, f2p))

    return locals()
    
