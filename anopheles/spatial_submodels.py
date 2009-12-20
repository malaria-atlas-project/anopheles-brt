import numpy as np
import cov_prior
from mahalanobis_covariance import *
import pymc as pm

def normalize_env(x, means, stds):
    x_norm = x.copy().reshape(-1,x.shape[-1])
    for i in xrange(2,x_norm.shape[1]):
        x_norm[:,i] -= means[i-2]
        x_norm[:,i] /= stds[i-2]
    return x_norm
                
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

def spatial_mahalanobis(x,y,dds,dde,amp,scale,val,vec,spat_frac,const_frac,symm=None):
    """
    The covariance of k + f_env(x) + f_spat, where the overall amplitude is fixed
    to 'amp'.
    """
    spat_amp = np.sqrt(spat_frac*amp**2)
    env_amp = np.sqrt((1-spat_frac-const_frac)*amp**2)
    const_amp = np.sqrt(const_frac*amp**2)
    spat_part = pm.gp.matern.geo_rad(x[:,:2],y[:,:2],amp=spat_amp,scale=scale,diff_degree=dds,symm=symm)
    env_part = mahalanobis_covariance(x[:,2:],y[:,2:],diff_degree=dde,amp=env_amp,val=val,vec=vec,symm=symm)
    
    out = spat_part+env_part+const_amp**2
    # if symm:
    #     np.testing.assert_almost_equal(out.max(),amp**2)
    #     
    #     import pylab as pl
    #     pl.clf()
    #     pl.subplot(1,2,1)
    #     pl.imshow(np.asarray(spat_part), interpolation='nearest')
    #     pl.title('spatial')
    #     pl.colorbar()
    #     pl.subplot(1,2,2)
    #     pl.imshow(np.asarray(env_part), interpolation='nearest')
    #     pl.title('environmental')
    #     pl.colorbar()
    #     
    #     from IPython.Debugger import Pdb
    #     Pdb(color_scheme='LightBG').set_trace() 

    return out
    
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

def lr_spatial_env(rl=200,**stuff):
    """A low-rank spatial-only model."""

    x_fr = normalize_env(stuff['full_x_fr'], stuff['env_means'], stuff['env_stds'])
    f2p = stuff['f2p']

    # ====================================================
    # = Covariance parameters of the environmental field =
    # ====================================================
    n_env = stuff['env_in'].shape[1]
    valpow = pm.Uniform('valpow',0,10,value=.9, observed=False)
    valbasemean = pm.Normal('valbasemean', 0, 1., value=0)
    valmean = pm.Lambda('valmean',lambda valpow=valpow, valbasemean=valbasemean : valbasemean + np.arange(n_env)*valpow)

    # valV = pm.Exponential('valV',1,value=.1)
    # val = pm.Normal('val',valmean,1./valV,value=np.ones(n_env)*2)
    # vals = [pm.Normal('val_%i'%i,valmean[i],1./valV,value=2) for i in xrange(n_env)]
    # val = pm.Lambda('val',lambda vals=vals: np.array(vals))

    expval = pm.Lambda('expval',lambda val=valmean: np.exp(val))    

    vec = cov_prior.OrthogonalBasis('vec',n_env,constrain=True)

    # =============================================
    # = Covariance parameter of the spatial field =
    # =============================================
    scale = pm.Exponential('scale',.1,value=.1)
    
    # =============================================================
    # = Parameters controlling relative sizes of field components =
    # =============================================================
    fracs = pm.Dirichlet('fracs', theta=np.repeat(2,3))
    const_frac=fracs[0]
    spat_frac=fracs[1]
    
    @pm.deterministic
    def C(val=expval,vec=vec,const_frac=const_frac,spat_frac=spat_frac,scale=scale):
        return pm.gp.FullRankCovariance(spatial_mahalanobis, dds=1.5, dde=1.5, amp=1.0, scale=scale,val=val, vec=vec, spat_frac=spat_frac, const_frac=const_frac)

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
    L_fr = pm.Lambda('L_fr',lambda U=U_fr: U.T, trace=False)

    # Evaluation of field at expert-opinion points
    init_val = np.ones(len(x_fr))*-.1
    f_fr = pm.MvNormalChol('f_fr', np.zeros(len(x_fr)), L_fr, value=init_val)

    @pm.deterministic(trace=False)
    def krige_wt(f_fr=f_fr, U_fr=U_fr):
        return pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    p = pm.Lambda('p', lambda x_fr=x_fr, C=C, krige_wt=krige_wt, means=stuff['env_means'], stds=stuff['env_stds'], f2p=f2p: LRP_norm(x_fr, C, krige_wt, means, stds, f2p))

    return locals()
    
