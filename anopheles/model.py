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

def threshold(f):
    return f > 0
    
def invlogit(f):
    return pm.flib.invlogit(f.ravel()).reshape(f.shape)

class F2pClosure(object):
    def __init__(self, f, f2p):
        self.f2p = f2p
        self.f = f
    def __call__(self, x):
        return self.f2p(self.f(x))
        

BinUBL = pm.stochastic_from_dist('BinUBL', bin_ubl_like, mv=True)
    
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
    p = pm.Lambda('p', lambda f=f, f2p=f2p: F2pClosure(f,f2p))
    
    # ==============
    # = Likelihood =
    # ==============

    if with_data:
        
        breaks, x, found, zero, others_found, multipoints = sites_as_ndarray(session, species)
        
        wherefound = np.where(found > 0)
        x_wherefound = x[wherefound]
        
        if len(env_variables)>0:
            env_x = np.array([extract_environment(n, x * 180./np.pi) for n in env_variables]).T
        else:
            env_x = np.empty((len(x),0))

        p_eval = p(np.hstack((x, env_x)))
        
        if multipoints:
            # FIXME: This will probably error out due to the new shape-checking in PyMC.
            data = pm.robust_init(BinUBL, 100, 'data', p_eval=p_eval, p_find=p_find, breaks=breaks, value=[found, others_found, zero], observed=True, trace=False)
        else:
            data = pm.Binomial('data', n=found+others_found+zero, p=p_eval*p_find, value=found, observed=True, trace=False)
            # data = pm.robust_init(pm.Binomial, 100, 'data', n=found+others_found+zero, p=p_eval*p_find, value=found, observed=True, trace=False)
    
    if with_eo:
        
        # ==============================
        # = Expert-opinion likelihoods =
        # ==============================        

        p_eval_in = pm.Lambda('p_eval_in', lambda p=p, x=pts_in, e=env_in: p(np.hstack((x, e))))
        p_eval_out = pm.Lambda('p_eval_out', lambda p=p, x=pts_out, e=env_out: p(np.hstack((x, e))))
        p_eval_eo = pm.Lambda('p_eval_eo', lambda p_eval_in=p_eval_in, p_eval_out=p_eval_out: np.concatenate((p_eval_in,p_eval_out)))
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
        
        constraint_dict = {}
        p_wherefound = np.ones(len(wherefound[0]))
        p_in_eo = np.ones(n_in)
        for k in constraint_fns.iterkeys():
            if k=='location':
                x_constraint = x
                x_constraint_eo = pts_eo
            else:
                x_constraint = env_dict[k]
                x_constraint_eo = env_dict_eo[k]
            
            # Make mighty sure that the constraint is satisfied at all datapoints.
            constraint_wherefound = constraint_fns[k](x=x_constraint[wherefound], p=p_wherefound)
            if constraint_wherefound>0:
                raise ValueError, 'Constraint function %s with input variable %s is violated the following locations where %s was found: \n%s' %\
                                    (constraint_fns[k].__name__, k, species[1], x_wherefound[np.where(constraint_wherefound>0)]*180./np.pi)
                                    
            # Make kind of sure that the constraint is satisfied in most of the expert opinion region.
            constraint_in_eo = constraint_fns[k](x=x_constraint_eo[:n_in], p=p_in_eo)
            if constraint_in_eo>0:
                warnings.warn('Constraint function %s with input variable %s is violated on %f of the expert opinion region for %s.' %\
                                    (constraint_fns[k].__name__, k, constraint_in_eo/float(n_in), species[1]))
            
            # Make sure the constraint doesn't get violated in the range of the species.
            constraint_dict[k] = Constraint(penalty_value = -1e100, logp=constraint_fns[k], doc="", name='%s_constraint'%k, parents={'x': x_constraint_eo, 'p': p_eval_eo})

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

def species_MCMC(session, species, spatial_submodel, db=None, **kwds):
    print 'Environment variables: ',kwds['env_variables']
    print 'Constraints: ',kwds['constraint_fns']
    print 'Spatial submodel: ',spatial_submodel.__name__
    print 'Species: ',species[1]
    if db is None:
        M=LatchingMCMC(make_model(session, species, spatial_submodel, **kwds), db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    else:
        M=LatchingMCMC(make_model(session, species, spatial_submodel, **kwds), db=db)
    bases = filter(lambda x: isinstance(x, OrthogonalBasis), M.stochastics)
    nonbases = filter(lambda x: True-isinstance(x, OrthogonalBasis), M.stochastics)
    am_scales = dict(zip(nonbases, [np.ones(nb.value.shape)*.001 for nb in nonbases]))
    M.use_step_method(pm.AdaptiveMetropolis, nonbases, delay=500000, scales=am_scales)
    for b in bases:
        M.use_step_method(GivensStepper, b)    
    if spatial_submodel.__name__ == 'lr_spatial':    
        M.use_step_method(MVNLRParentMetropolis, scalar_stochastics, M.f_fr, M.U, M.piv, M.rl)
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