# Copyright (C) 2009  Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib
matplotlib.use('Qt4Agg')

from mpl_toolkits import basemap
b = basemap.Basemap(0,0,1,1)
import pymc as pm
import numpy as np
from query_to_rec import *
from env_data import *
from mahalanobis_covariance import mahalanobis_covariance
from map_utils import multipoly_sample
from mapping import *
from spatial_submodels import *
from constraints import *
from cov_prior import GivensStepper, OrthogonalBasis
from utils import bin_ubls
import datetime
import warnings
import shapely

__all__ = ['make_model', 'species_MCMC', 'probability_traces','potential_traces','threshold','invlogit']

def bin_ubl_like(x, p_eval, p_find, breaks):
    "The ubl likelihood function, document this."
    return bin_ubls(x[0], x[0]+x[1]+x[2], p_find, breaks, p_eval)

def identity(x):
    return x

def threshold(f):
    return f > 0
    
def invlogit(f):
    return pm.flib.invlogit(f.ravel()).reshape(f.shape)

class LRP(object):
    """A closure that can evaluate a low-rank field."""
    def __init__(self, f, x_fr, C, krige_wt, f2p):
        self.f = f
        self.x_fr = x_fr
        self.C = C
        self.krige_wt = krige_wt
        self.f2p = f2p
    def __call__(self, x, f2p=None, offdiag = None):
        if f2p is None:
            f2p = self.f2p
        if offdiag is None:
            x_ = x.reshape(-1,x.shape[-1])
            x__ = x_[:,:2].reshape(x.shape[:-1]+(2,))
            offdiag = self.C(x__,self.x_fr)
        out = np.dot(np.asarray(offdiag), self.krige_wt).reshape(x.shape[:-1]) + self.f(x)
        return f2p(out)

class LRP_norm(LRP):
    """
    A closure that can evaluate a low-rank field.

    Normalizes the third argument onward.
    """
    def __init__(self, f, x_fr, C, krige_wt, means, stds, f2p):
        LRP.__init__(self, f, x_fr, C, krige_wt, f2p)
        self.means = means
        self.stds = stds

    def __call__(self, x, f2p=None, offdiag=None):
        x_norm = normalize_env(x, self.means, self.stds)
        return LRP.__call__(self, x_norm.reshape(x.shape),f2p,offdiag)
        

BinUBL = pm.stochastic_from_dist('BinUBL', bin_ubl_like, mv=True)

class SubsetMetropolis(pm.Metropolis):

    def __init__(self, stochastic, index, interval, *args, **kwargs):
        self.index = index
        self.interval = interval
        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)

    def propose(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        newval = self.stochastic.value.copy()
        newval[self.index:self.index+self.interval] += np.random.normal(size=self.interval) * self.proposal_sd[self.index:self.index+self.interval]*self.adaptive_scale_factor
        self.stochastic.value = newval
    
def make_model(session, species, spatial_submodel, with_eo = True, with_data = True, env_variables = (), constraint_fns={}, n_in=1000, n_out=1000, f2p=threshold):
    """
    Generates a PyMC probability model with a plug-in spatial submodel.
    The likelihood and expert-opinion layers are common.
    
    Constraints are hard expert opinions. The keys should be either
    'location' or members of env_variables. The values should be functions
    that take two arguments labelled 'x' and 'p'. The first will be either 
    a 2xn array of locations or a length-n array of environmental variable 
    values. The second will be a length-n array of evaluations of p. The 
    return value should be a boolean. If True, p is acceptable; if False, 
    it is not.
    
    These constraints will be exposed with names '%s_constraint'%key. They
    can be opened and closed as normal.
    
    Note that constraints will not be created unless with_eo=True.
    """

    # =========
    # = Query =
    # =========
    
    pts_in, pts_out = sample_eo(session, species, n_in, n_out)
    
    # ========================
    # = Environmental inputs =
    # ========================
    
    if len(env_variables)>0:
        env_in = np.array([extract_environment(n, pts_in * 180./np.pi) for n in env_variables]).T
        env_out = np.array([extract_environment(n, pts_out * 180./np.pi) for n in env_variables]).T
        env_means = np.array([np.mean(np.concatenate((env_in[:,i], env_out[:,i]))) for i in xrange(len(env_variables))])
        env_stds = np.array([np.std(np.concatenate((env_in[:,i], env_out[:,i]))) for i in xrange(len(env_variables))])
    else:
        env_in = np.empty((len(pts_in),0))
        env_out = np.empty((len(pts_out),0))
        env_means = []
        env_stds = []
    
    # ==========
    # = Priors =
    # ==========
    
    p_find = pm.Uniform('p_find',0,1)
    # p_find = pm.Uniform('p_find',.9,1)
    spatial_variables = spatial_submodel(**locals())
    f = spatial_variables['f']
        
    # =======================
    # = Correlated 'nugget' =
    # =======================
    scale = pm.Uniform('scale',0,.5,value=.05,observed=False)
    
    @pm.deterministic
    def C(scale=scale):
        return pm.gp.FullRankCovariance(pm.gp.exponential.geo_rad, amp=1, scale=scale)
    
    # diff_degree = pm.Uniform('diff_degree',0,3,value=2)
    # @pm.deterministic
    # def C(scale=scale, diff_degree=diff_degree):
    #     return pm.gp.FullRankCovariance(pm.gp.matern.geo_rad, amp=1, scale=scale, diff_degree=diff_degree)

    full_x_in = np.hstack((pts_in, env_in))
    full_x_out = np.hstack((pts_out, env_out))
    full_x_eo = np.vstack((full_x_in, full_x_out))
    x_eo = np.vstack((pts_in, pts_out))
    
    x_fr = x_eo[::5]
    full_x_fr = full_x_eo[::5]
    
    f_eval = f(normalize_env(full_x_fr, env_means, env_stds))

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
    init_val = np.ones(len(x_fr))*.1
    init_val[:len(pts_in)] = 1
    init_val = None
    
    f_fr = pm.MvNormalChol('f_fr', f_eval, L_fr, value=init_val)
    f_fr_ = f_fr - f_eval
    
    @pm.deterministic(trace=False)
    def krige_wt(f_fr=f_fr_, U_fr=U_fr, f=f_eval):
        if U_fr.shape == f_fr.shape*2:
            return pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)
        else:
            return None
    
    p = pm.Lambda('p', lambda f=f, x_fr=x_fr, C=C, krige_wt=krige_wt, means=env_means, stds=env_stds, f2p=f2p: LRP_norm(f, x_fr, C, krige_wt, means, stds, f2p))
    
    if with_data:
        
        # ==============
        # = Likelihood =
        # ==============
        
        breaks, x, found, zero, others_found, multipoints = sites_as_ndarray(session, species)
        
        wherefound = np.where(found > 0)
        x_wherefound = x[wherefound]
        
        if len(env_variables)>0:
            env_x = np.array([extract_environment(n, x * 180./np.pi) for n in env_variables]).T
        else:
            env_x = np.empty((len(x),0))
        env_x_wherefound = env_x[wherefound]            
    
        p_eval = pm.Lambda('p_eval', lambda p=p, od=C(x,x_fr): p(np.hstack((x, env_x)), offdiag=np.asarray(od)), trace=False)        
        f_eval_wherefound = pm.Lambda('f_eval_wherefound', lambda p=p, od=C(x_wherefound,x_fr): p(np.hstack((x_wherefound, env_x_wherefound)), offdiag=np.asarray(od), f2p=identity), trace=False)
    
        constraint_dict = {'data': Constraint(penalty_value = -1e100, logp=lambda f: -np.sum(f*(f<0)), doc="", name='data_constraint', parents={'f':f_eval_wherefound})}
    
        if multipoints:
            # FIXME: This will probably error out due to the new shape-checking in PyMC.
            # data = pm.robust_init(BinUBL, 100, 'data', p_eval=p_eval, p_find=p_find, breaks=breaks, value=[found, others_found, zero], observed=True, trace=False)
            raise NotImplementedError
        else:
            pass
            # ===================================================================================
            # = NB: Data is now created by species_MCMC after all constraints have been closed! =
            # ===================================================================================
            # data = pm.Binomial('data', n=found+others_found+zero, p=p_eval*p_find, value=found, observed=True, trace=False)
            # data = pm.robust_init(pm.Binomial, 100, 'data', n=found+others_found+zero, p=p_eval*p_find, value=found, observed=True, trace=False)
    
    if with_eo:
        
        # ==============================
        # = Expert-opinion likelihoods =
        # ==============================
        
        f_eval_in = pm.Lambda('f_eval_in', lambda p=p, od=C(pts_in, x_fr): p(full_x_in, f2p=identity, offdiag=od), trace=False)
        f_eval_out = pm.Lambda('f_eval_out', lambda p=p, od=C(pts_out, x_fr): p(full_x_out, f2p=identity, offdiag=od), trace=False)   
        f_eval_eo = pm.Lambda('f_eval_eo', lambda f_eval_in=f_eval_in, f_eval_out=f_eval_out: np.concatenate((f_eval_in,f_eval_out)), trace=False)
    
        p_eval_in = pm.Lambda('p_eval_in', lambda f=f_eval_in, f2p=f2p: f2p(f), trace=False)
        p_eval_out = pm.Lambda('p_eval_out', lambda f=f_eval_out, f2p=f2p: f2p(f), trace=False)        
        p_eval_eo = pm.Lambda('p_eval_eo', lambda p_eval_in=p_eval_in, p_eval_out=p_eval_out: np.concatenate((p_eval_in,p_eval_out)), trace=False)
    
        in_prob = pm.Lambda('in_prob', lambda p_eval = p_eval_in: np.mean(p_eval)*.9999+.00005)
        out_prob = pm.Lambda('out_prob', lambda p_eval = p_eval_out: np.mean(p_eval)*.9999+.00005)    
        
        alpha_out = pm.Uniform('alpha_out',0,1)
        beta_out = pm.Uniform('beta_out',1,10)
        alpha_in = pm.Uniform('alpha_in',1,10)
        beta_in = pm.Uniform('beta_in',0,1)
            
        @pm.potential
        def out_factor(a=alpha_out, b=beta_out, p=out_prob):
            return pm.beta_like(p,a,b)
        
        @pm.potential
        def in_factor(a=alpha_in, b=beta_in, p=in_prob):
            return pm.beta_like(p,a,b)
        
        # ========================
        # = Hard expert opinions =
        # ========================
    
        env_eo = np.vstack((env_in, env_out))
        pts_eo = np.vstack((pts_in,pts_out))
        env_dict = dict(zip(env_variables,env_x.T))
        env_dict_eo = dict(zip(env_variables,env_eo.T))
        
        f_wherefound = np.ones(len(wherefound[0]))
        f_in_eo = np.ones(n_in)
        for k in constraint_fns.iterkeys():
            if k=='location':
                x_constraint = x
                x_constraint_eo = pts_eo
            else:
                x_constraint = env_dict[k]
                x_constraint_eo = env_dict_eo[k]
            
            # Make mighty sure that the constraint is satisfied at all datapoints.
            constraint_wherefound = constraint_fns[k](x=x_constraint[wherefound], f=f_wherefound)
            if constraint_wherefound>0:
                raise ValueError, 'Constraint function %s with input variable %s is violated the following locations where %s was found: \n%s' %\
                                    (constraint_fns[k].__name__, k, species[1], x_wherefound[np.where(constraint_wherefound>0)]*180./np.pi)
                                    
            # Make kind of sure that the constraint is satisfied in most of the expert opinion region.
            constraint_in_eo = constraint_fns[k](x=x_constraint_eo[:n_in], f=f_in_eo)
            if constraint_in_eo>0:
                warnings.warn('Constraint function %s with input variable %s is violated on %f of the expert opinion region for %s.' %\
                                    (constraint_fns[k].__name__, k, constraint_in_eo/float(n_in), species[1]))
            
            # Make sure the constraint doesn't get violated in the range of the species.
            constraint_dict[k] = Constraint(penalty_value = -1e100, logp=constraint_fns[k], doc="", name='%s_constraint'%k, parents={'x': x_constraint_eo, 'f': f_eval_eo})

    out = locals()
    out.update(spatial_variables)
    return out

def probability_traces(M, pos_or_neg = True):
    "Plots traces of the probability of presence at observation locations."
    if pos_or_neg:
        x = M.x[np.where(M.found > 0)]
    else:
        x = M.x[np.where(M.found == 0)]
    vals = []
    for i in xrange(M._cur_trace_index):
        p = M.trace('p')[i:i+1][0]
        vals.append(p(x))
    return np.array(vals)

def potential_traces(M, in_or_out = 'in'):
    "Traces of the 'means' of the EO factor potentials."
    a = M.trace('alpha_'+in_or_out)[:]
    b = M.trace('beta_'+in_or_out)[:]    
    
    import pylab as pl
    pl.plot(a/(a+b))

def species_stepmethods(M, interval=None):
    """
    Adds appropriate step methods to M.
    """
    bases = filter(lambda x: isinstance(x, OrthogonalBasis), M.stochastics)
    nonbases = set(filter(lambda x: True-isinstance(x, OrthogonalBasis), M.stochastics))

    nonbases.discard(M.f_fr)
    if interval is not None:
        for i in xrange(0,len(M.f_fr.value),interval):
            M.use_step_method(SubsetMetropolis, M.f_fr, i, interval)
    else:
        M.use_step_method(pm.AdaptiveMetropolis, M.f_fr, scales={M.f_fr: .0001*np.ones(M.f_fr.value.shape)})

    nonbases = list(nonbases)
    am_scales = dict(zip(nonbases, [np.ones(nb.value.shape)*.001 for nb in nonbases]))
    # M.use_step_method(pm.AdaptiveMetropolis, nonbases, delay=500000, scales=am_scales)

    for b in bases:
        M.use_step_method(GivensStepper, b)    

def species_MCMC(session, species, spatial_submodel, db=None, **kwds):
    print 'Environment variables: ',kwds['env_variables']
    print 'Constraints: ',kwds['constraint_fns']
    print 'Spatial submodel: ',spatial_submodel.__name__
    print 'Species: ',species[1]

    model = make_model(session, species, spatial_submodel, **kwds)
    if db is None:
        M=LatchingMCMC(model, db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    else:
        M=LatchingMCMC(model, db=db)
    species_stepmethods(M, interval=5)

    print 'Attempting to satisfy constraints'
    M.isample(1)
    
    print 'Done, creating data object and returning'
    found = model['found']
    others_found = model['others_found']
    zero = model['zero']
    p_eval = model['p_eval']
    p_find = model['p_find']
    M.data = pm.Binomial('data', n=found+others_found+zero, p=p_eval*p_find, value=found, observed=True, trace=False)            
    
    del M.step_methods
    M._sm_assigned = False
    M.step_method_dict = {}
    for s in M.stochastics:
        M.step_method_dict[s] = []
    
    species_stepmethods(M, interval=5)
    # bases = filter(lambda x: isinstance(x, OrthogonalBasis), M.stochastics)
    # for b in bases:
    #     M.use_step_method(GivensStepper, b)    
    # 
    # M.use_step_method(pm.AdaptiveMetropolis, M.f_fr, scales={M.f_fr: .0001*np.ones(M.f_fr.value.shape)})
    return M

def mean_response_samples(M, axis, n, burn=0, thin=1):
    pts_in = M.pts_in
    pts_out = M.pts_out
    pts_eo = np.vstack((pts_in, pts_out))
    
    x_disp = np.linspace(pts_eo[:,axis].min(), pts_eo[:,axis].max(), n)
    
    pts_plot = np.tile(pts_eo.T, n).T
    pts_plot[:,axis] = np.repeat(x_disp, len(pts_eo))
    
    outs = []
    for p in M.trace('p')[:][burn::thin]:
        out = np.empty(n)
        p_eval = p(pts_plot)
        for i in xrange(n):
            out[i] = np.mean(p_eval[i*len(pts_eo):(i+1)*len(pts_eo)])
        outs.append(out)
            
    return x_disp, outs
    
def initialize_by_eo(M):    
    new_val_fr = np.empty(M.rl)
    new_val_fr[:] = -5
    new_val_fr[np.where(M.piv.value[:M.rl]<M.pts_in.shape[0])] = 10
    
    new_val_d = np.dot(M.U.value[:,M.rl:].T,np.linalg.solve(M.U.value[:,:M.rl].T, new_val_fr))
    
    new_val = np.empty(len(M.f_eo.value))        
    new_val[M.piv.value[:M.rl]] = new_val_fr
    new_val[M.piv.value[M.rl:]] = new_val_d
    M.f_eo.value = new_val
    
    
# TODO: Evaluation metrics: kappa etc.
# TODO: Figure out how to assess convergence even though there's degeneracy: can exchange axis labels in covariance prior and no problem.
# TODO: Actually if there's convergence without reduction, you're fine.