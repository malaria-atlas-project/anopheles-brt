import numpy as np
import cov_prior
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step','MvNormalLR','lr_spatial','MVNLRMetropolis']

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
        
        # Verbosity flag
        self.verbose = verbose
        
        self.accepted = 0
        self.rejected = 0
        self.mvnlr = mvnlr
        self.piv = mvnlr.parents['piv']
        self.U = mvnlr.parents['U']
        self.rl = mvnlr.parents['rl']
        
        stochastic = [mvnlr]
        # Initialize superclass
        pm.StepMethod.__init__(self, stochastic, verbose, tally)
        
        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += ['accepted', 'rejected', '_trace_count', '_current_iter', 'C', 'proposal_sd',
        '_proposal_deviate', '_trace']
        self._tuning_info = ['C']
        
        self.proposal_sd = None
        
        # Number of successful steps before the empirical covariance is computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy
        
        # Call methods to initialize
        self.isdiscrete = {mvnlr: False}
        self.dim = self.rl
        self._slices = {mvnlr: slice(0,self.rl)}
        self.set_cov(cov, scales)
        self.updateproposal_sd()
        
        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy
        # sampling can be done during warm-up period.
        self._trace_count = 0
        self._current_iter = 0
        
        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []
        
        if self.verbose >= 1:
            print "Initialization..."
            print 'Dimension: ', self.dim
            print "C_0: ", self.C
            print "Sigma: ", self.proposal_sd
    
    @classmethod
    def competence(cls,stochastic):
        if isinstance(stochastic, MvNormalLR):
            return 3
        else:
            return 0
    
    def set_cov(self, cov=None, scales={}, trace=2000, scaling=50):
        """
        Define C, the jump distributioin covariance matrix.

        Return:
            - cov,  if cov != None
            - covariance matrix built from the scales dictionary if scales!=None
            - covariance matrix estimated from the stochastics last trace values.
            - covariance matrix estimated from the stochastics value, scaled by
                scaling parameter.
        """

        if cov is not None:
            self.C = cov
        elif scales:
            # Get array of scales
            ord_sc = self.order_scales(scales)
            # Scale identity matrix
            self.C = np.eye(self.dim)*ord_sc
        else:
            try:
                a = self.trace2array(-trace, -1)
                nz = a[:, 0]!=0
                self.C = np.cov(a[nz, :], rowvar=0)
            except:
                ord_sc = []
                for s in self.stochastics:
                    this_value = abs(np.ravel(s.value[self.piv.value[:self.rl]]))
                    if not this_value.any():
                        this_value = [1.]
                    for elem in this_value:
                        ord_sc.append(elem)
                # print len(ord_sc), self.dim
                for i in xrange(len(ord_sc)):
                    if ord_sc[i] == 0:
                        ord_sc[i] = 1
                self.C = np.eye(self.dim)*ord_sc/scaling

            
    def propose(self):
        
        # from IPython.Debugger import Pdb
        # Pdb(color_scheme='Linux').set_trace()   
        
        piv = self.piv.value
        v = self.mvnlr.value[piv[:self.rl]]
        jump = np.dot(self.proposal_sd, np.random.normal(size=self.proposal_sd.shape[0]))
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
