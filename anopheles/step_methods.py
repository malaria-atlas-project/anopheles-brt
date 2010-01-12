import pymc as pm
import numpy as np
import warnings
from constrained_mvn_sample import cmvns_l
import time

def value_and_maybe_copy(v):
    v = pm.utils.value(v)
    if isinstance(v, np.ndarray):
        return v.copy('F')
    return v        
    
def union(sets):
    out = set()
    for s in sets:
        out |= s
    return out

class CMVNLStepper(pm.StepMethod):
    """
    Arguments:
        - f : Multivariate normal
        - g : U^{-T} f, a deterministic
        - U : Upper triangular Cholesky factor of covariance of f.
        - likelihood_offdiags: C(xp, x) U^{-1} for xp in the set of evaluation locations
            that don't correspond to hard constraints
        - constraint_offdiags: C(xp, x) U^{-1} for xp in the set of evaluation locations
            that do correspond to hard constraints
        - constraint_signs: Whether f has to be positive or negative at the xp's in 
            constraint_offdiags.
    """
    
    def __init__(self, f, g, U, likelihood_offdiags, constraint_offdiags, constraint_signs):
        self.f = f
        self.g = g
        self.U = U
        self.n = len(self.g.value)
        self.likelihood_offdiags = likelihood_offdiags
        self.constraint_offdiags = constraint_offdiags
        self.all_offdiags = list(self.likelihood_offdiags) + list(self.constraint_offdiags)
        self.constraint_signs = constraint_signs
        self.adaptive_scale_factor = np.ones(self.n)
        self.accepted = np.zeros(self.n)
        self.rejected = np.zeros(self.n)
        
        self.likelihood_children = union([pm.extend_children(od.children) for od in self.likelihood_offdiags])
        
        pm.StepMethod.__init__(self, f)

    def get_bounds(self, i):
        # Linear constraints.
        lb = -4
        ub = 4
        rhs = {}
        g = self.g.value
        for j,od in enumerate(self.constraint_offdiags):
            rhs[od] = self.rhs[od].copy()
            coef = np.asarray(pm.utils.value(od))[:,i].ravel()
            rhs[od] -= coef * g[i]
            
            where_coef_neg = np.where(coef<0)
            where_coef_pos = np.where(coef>0)
            
            lolims = -rhs[od][where_coef_pos] / coef[where_coef_pos]
            uplims = -rhs[od][where_coef_neg] / coef[where_coef_neg]
            
            if self.constraint_signs[j] == -1:
                tmp = uplims
                uplims = lolims
                lolims = tmp
            
            lb = np.hstack((lb, lolims)).max()
            ub = np.hstack((ub, uplims)).min()
        return lb, ub, rhs

    def step(self):
        t = [0, 0, 0]
        self.last_accepted = True
        cv = {}
        
        # The right-hand sides for the linear constraints
        self.rhs = dict(zip(self.constraint_offdiags, 
                            [np.asarray(np.dot(pm.utils.value(od), self.g.value)).ravel() for od in self.constraint_offdiags]))
        
        for i in xrange(self.n):
            t1 = time.time()
            # Make copies of the f_evals' values now, they won't be available
            # once f_fr changes.
            if self.last_accepted:
                for od in self.all_offdiags:
                    for c in od.children:
                        cv[c] = c.value.copy()
            t[0] += time.time() - t1

            # Jump an element of g.
            lb, ub, rhs = self.get_bounds(i)
                
            g = self.g.value.copy()
            newgi = pm.rtruncnorm(0,1,lb,ub)
            dg = newgi-g[i]
            g[i]=newgi
            
            for od in self.constraint_offdiags:
                rhs[od] += np.asarray(pm.utils.value(od))[:,i].ravel() * g[i]

            lpl = pm.utils.logp_of_set(self.likelihood_children)
            
            t1 = time.time()
            # Record change in f.
            self.f.value = self.f.value + np.asarray(pm.utils.value(self.U)[i,:]).ravel()*dg
            self.g._value.force_cache(g)
            
            for j,od in enumerate(self.constraint_offdiags):
                # The children of the offdiags are just the f_evals.
                for c in od.children:
                    c._value.force_cache(cv[c] + np.asarray(od.value[:,i]).ravel()*dg)
                    if np.any(c.value*self.constraint_signs[j]<0):
                        raise ValueError, 'Constraint broken!'
            
            for od in self.likelihood_offdiags:
                # The children of the offdiags are just the f_evals.
                for c in od.children:
                    c._value.force_cache(cv[c] + np.asarray(od.value[:,i]).ravel()*dg)
            
            t[1] += time.time() - t1
            
            t1 = time.time()
            try:
                lpl_p = pm.utils.logp_of_set(self.likelihood_children)
            except pm.ZeroProbability:
                self.reject(i)
                continue
            
            if np.log(np.random.random()) < lpl_p - lpl:
                self.accepted[i] += 1
                self.last_accepted = True
                self.rhs = rhs
            else:
                self.reject(i)
            t[2] += time.time() - t1
        print t

    def reject(self, i):
        self.f.revert()
        self.rejected[i] += 1
        self.last_accepted = False
        
            
class DelayedMetropolis(pm.Metropolis):

    def __init__(self, stochastic, sleep_interval=1, *args, **kwargs):
        self._index = -1        
        self.sleep_interval = sleep_interval
        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)

    def step(self):
        self._index += 1
        if self._index % self.sleep_interval == 0:
            pm.Metropolis.step(self)    

class KindOfConditional(pm.Metropolis):

    def __init__(self, stochastic, cond_jumper):
        pm.Metropolis.__init__(self, stochastic)
        self.stochastic = stochastic
        self.cond_jumper = cond_jumper
        
    def propose(self):        
        if self.cond_jumper.value is None:
            pass
        else:
            self.stochastic.value = self.cond_jumper.value()
        
    def hastings_factor(self):
        if self.cond_jumper.value is not None:
            for_factor = pm.mv_normal_chol_like(self.stochastic.value, self.cond_jumper.value.M_cond, self.cond_jumper.value.L_cond)
            back_factor = pm.mv_normal_chol_like(self.stochastic.last_value, self.cond_jumper.value.M_cond, self.cond_jumper.value.L_cond)
            return back_factor - for_factor
        else:
            return 0.
                

class MVNPriorMetropolis(pm.Metropolis):

    def __init__(self, stochastic, L):
        self.stochastic = stochastic
        self.L = L
        pm.Metropolis.__init__(self, stochastic)
        self.adaptive_scale_factor = .001

    def propose(self):
        dev = pm.rmv_normal_chol(np.zeros(self.stochastic.value.shape), self.L.value)
        dev *= self.adaptive_scale_factor
        self.stochastic.value = self.stochastic.value + dev

class SubsetMetropolis(DelayedMetropolis):

    def __init__(self, stochastic, index, interval, sleep_interval=1, *args, **kwargs):
        self.index = index
        self.interval = interval
        DelayedMetropolis.__init__(self, stochastic, sleep_interval, *args, **kwargs)
        self.adaptive_scale_factor = .01

    def propose(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        newval = self.stochastic.value.copy()
        newval[self.index:self.index+self.interval] += np.random.normal(size=self.interval) * self.proposal_sd[self.index:self.index+self.interval]*self.adaptive_scale_factor
        self.stochastic.value = newval

def gramschmidt(v):
    m = np.eye(len(v))[:,:-1]
    for i in xrange(len(v)-1):
        m[:,i] -= v*v[i]
        for j in xrange(0,i):
            m[:,i] -= m[:,j]*np.dot(m[:,i],m[:,j])
        m[:,i] /= np.sqrt(np.sum(m[:,i]**2))
    return m

class RayMetropolis(DelayedMetropolis):
    """
    Approximately Gibbs samples along a randomly-selected ray.
    Always has the option to maintain current state.
    """
    def __init__(self, stochastic, sleep_interval=1):
        DelayedMetropolis.__init__(self, stochastic, sleep_interval)
        self.v = 1./self.stochastic.parents['tau']
        self.m = self.stochastic.parents['mu']
        self.n = len(self.stochastic.value)
        self.f_fr = None
        self.other_children = set([])
        for c in self.stochastic.extended_children:
            if c.__class__ is pm.MvNormalChol:
                self.f_fr = c
            else:
                self.other_children.add(c)
        if self.f_fr is None:
            raise ValueError, 'No f_fr'
        
    def step(self):
        self._index += 1
        if self._index % self.sleep_interval == 0:
            
            v = pm.value(self.v)
            m = pm.value(self.m)
            val = self.stochastic.value
            lp = pm.logp_of_set(self.other_children)
        
            # Choose a direction along which to step.
            dirvec = np.random.normal(size=self.n)
            dirvec /= np.sqrt(np.sum(dirvec**2))
        
            # Orthogonalize
            orthoproj = gramschmidt(dirvec)
            scaled_orthoproj = v*orthoproj.T
            pck = np.dot(dirvec, scaled_orthoproj.T)
            kck = np.linalg.inv(np.dot(scaled_orthoproj,orthoproj))
            pckkck = np.dot(pck,kck)

            # Figure out conditional variance
            condvar = np.dot(dirvec, dirvec*v) - np.dot(pck, pckkck)
            # condmean = np.dot(dirvec, m) + np.dot(pckkck, np.dot(orthoproj.T, (val-m)))
        
            # Compute slice of log-probability surface
            tries = np.linspace(-4*np.sqrt(condvar), 4*np.sqrt(condvar), 501)
            lps = 0*tries
        
            for i in xrange(len(tries)):
                new_val = val + tries[i]*dirvec
                self.stochastic.value = new_val
                try:
                    lps[i] = self.f_fr.logp + self.stochastic.logp
                except:
                    lps[i] = -np.inf              
            if np.all(np.isinf(lps)):
                raise ValueError, 'All -inf.'
            lps -= pm.flib.logsum(lps[True-np.isinf(lps)])          
            ps = np.exp(lps)
        
            index = pm.rcategorical(ps)
            new_val = val + tries[index]*dirvec
            self.stochastic.value = new_val
            
            try:
                lpp = pm.logp_of_set(self.other_children)
                if np.log(np.random.random()) < lpp - lp:
                    self.accepted += 1
                else:
                    self.stochastic.value = val
                    self.rejected += 1
                    
            except pm.ZeroProbability:
                self.stochastic.value = val
                self.rejected += 1
        self.logp_plus_loglike
        
