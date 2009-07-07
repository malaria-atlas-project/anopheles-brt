import numpy as np
import cov_prior
import pymc as pm

__all__ = ['spatial_hill','hill_fn','hinge','step','LRGP','LRRealization','LRGPMetropolis','lr_spatial']

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
class LRRealization(object):
    def __init__(self, val, x_fr, U, C):
        self.x_fr = x_fr
        self.U = U
        self.C = C
        self.val = val
        self.shift(x_fr, U)        
        

    def shift(self, x_fr, U):
        "Shifts to a new x_fr or U."
        self.U = U
        self.x_fr = x_fr
        self.krige_wt = pm.gp.trisolve(self.U,pm.gp.trisolve(self.U,self.val,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)

    def call_untrans(self, x, U=None, x_fr=None):
        f_out = None

        # Possibly shift
        if x_fr is not None:
            if np.any(x_fr != self.x_fr):
                self.val = self.call_untrans(x_fr, U, x_fr)                
                f_out = self.shift(x_fr, U)

        if f_out is None:
            f_out = np.asarray(np.dot(self.krige_wt,self.C(self.x_fr,x)))
            
        return f_out

    def __call__(self, x, U=None, x_fr=None):
        f_out = self.call_untrans(x, U, x_fr)
        return pm.invlogit(f_out.ravel()).reshape(x.shape[:-1])


def lrgp_lp(value, x, U, rl, C):
    return pm.mv_normal_chol_like(value.call_untrans(x,U,x), np.zeros(rl), U)

class LRGP(pm.Stochastic):
    def __init__(self, name, x_fr, U, C):
        self.rl = x_fr.value.shape[0]
        pm.Stochastic.__init__(self, doc='', 
            name=name, 
            parents={'x': x_fr, 'U': U, 'rl': self.rl, 'C': C}, 
            logp=lrgp_lp, 
            value = LRRealization(np.zeros(x_fr.value.shape[:-1]), x_fr.value, U.value, C.value), 
            dtype=np.dtype('object'))

class LRGPMetropolis(pm.AdaptiveMetropolis):
    def __init__(self, lrgp, cov=None, delay=1000, scales=None, interval=200, greedy=True, shrink_if_necessary=False, verbose=0, tally=False):
        
        # Verbosity flag
        self.verbose = verbose
        
        self.accepted = 0
        self.rejected = 0
        self.lrgp = lrgp
        
        stochastic = [lrgp]
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
        self.isdiscrete = {lrgp: False}
        self.dim = lrgp.rl
        self._slices = {lrgp: slice(0,lrgp.rl)}
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
        if isinstance(stochastic, LRGP):
            return 3
        else:
            return 0
    
    def set_cov(self, cov=None, scales={}, trace=2000, scaling=50):
        """Define C, the jump distributioin covariance matrix.

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
                    this_value = abs(np.ravel(s.value.val))
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
        s = self.lrgp
        v = s.value
        jump = np.dot(self.proposal_sd, np.random.normal(size=self.proposal_sd.shape[0]))
        if self.verbose > 2:
            print 'Jump :', jump
        # Update each stochastic individually.
        new_val = v.val + jump
        s.value = LRRealization(new_val, v.x_fr, v.U, v.C)

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
    x_eo = np.vstack((pts_in, pts_out))

    @pm.deterministic
    def C(amp=amp,scale=scale,diff_degree=diff_degree):
        return pm.gp.Covariance(mod_matern, amp=amp, scale=scale, diff_degree=diff_degree)

    @pm.deterministic(trace=False)
    def x_and_U(C=C, rl=rl, x=x_eo):
        d = C.cholesky(x, rank_limit=rl, apply_pivot=False)
        piv = d['pivots']
        U = d['U']
        return x[piv[:rl]], U[:rl,:rl]

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

    f = LRGP('f', x_fr, U, C)
    
    return locals()            
