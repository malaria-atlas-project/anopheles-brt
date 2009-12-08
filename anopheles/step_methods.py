import pymc as pm
import numpy as np
from constrained_mvn_sample import cmvns_l

class CMVNLStepper(pm.StepMethod):
    """
    A constrained multivariate normal stepper with likelihood.
    Metropolis samples self.stochastic under the constraint that B*self.stochastic < y,
    with likelihood term corresponding to n_neg negative observations independently 
    with probabilities p_find if Bl*cur_val > 0, else 0.
    """
    
    def __init__(self, stochastic, B, y, Bl, n_neg, p_find, pri_S, pri_M, n_cycles=1, pri_S_type='square'):
        self.stochastic = stochastic
        self.cvmns_l_params = [B, y, Bl, n_neg, p_find, pri_S, pri_M, n_cycles, pri_S_type]
        pm.StepMethod.__init__(self, stochastic)
        
    def step(self):
        self.stochastic.value, self.accepted, self.rejected = \
             cmvns_l(self.stochastic.value, *tuple([pm.utils.value(v) for v in self.cvmns_l_params]))

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
        
