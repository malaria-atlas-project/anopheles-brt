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
from mpl_toolkits import basemap
b = basemap.Basemap(0,0,1,1)
import pymc as pm
import numpy as np
from step_methods import *
from query_to_rec import *
from env_data import *
from mahalanobis_covariance import *
from map_utils import multipoly_sample
from mapping import *
from spatial_submodels import *
from constraints import *
from cov_prior import GivensStepper, OrthogonalBasis
import datetime
import warnings
import shapely
import tables as tb

__all__ = ['make_model', 'species_MCMC', 'threshold','invlogit', 'restore_species_MCMC']

def identity(x):
    return x

def threshold(f):
    return f > 0
    
def invlogit(f):
    return pm.flib.invlogit(f.ravel()).reshape(f.shape)
    
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
    
    # Evaluate the environmental surfaces at the inducing points.
    if len(env_variables)>0:
        env_in = np.array([extract_environment(n, pts_in * 180./np.pi) for n in env_variables]).T
        env_out = np.array([extract_environment(n, pts_out * 180./np.pi) for n in env_variables]).T
        # Record the means and standard deviations, because the surfaces will be scaled and shifted
        # according to those before input into the fields.
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
    
    p_find = pm.Uniform('p_find',0,1, doc="Probability of finding a species within its range.")

    # The full_ prefix implies that 'x' is the concatenation of lon,lat, and all environmental surfaces.
    # Just 'x' means it's just lon,lat (in radians).
    # 'env_' means it's just the environmental surfaces.
    full_x_in = np.hstack((pts_in, env_in))
    full_x_out = np.hstack((pts_out, env_out))
    full_x_eo = np.vstack((full_x_in, full_x_out))
    full_x_in_n = normalize_env(full_x_in, env_means, env_stds)
    full_x_out_n = normalize_env(full_x_out, env_means, env_stds)    
    full_x_eo_n = normalize_env(full_x_eo, env_means, env_stds)
    x_eo = np.vstack((pts_in, pts_out))
    
    # The '_fr' suffix means 'on the inducing points'.
    x_fr = x_eo[::5]
    full_x_fr = full_x_eo[::5]
    full_x_fr_n = normalize_env(full_x_fr, env_means, env_stds)

    # ============================
    # = Call to spatial submodel =
    # ============================
    spatial_variables = spatial_submodel(**locals())
    p = spatial_variables['p']
    f_fr = spatial_variables['f_fr']
    val = spatial_variables['val']
    vec = spatial_variables['vec']
    C = spatial_variables['C']
    
    if with_data:
        # ==============
        # = Likelihood =
        # ==============
        
        # Read in the data, and split it up.
        breaks, x, found, zero, others_found, multipoints = sites_as_ndarray(session, species)
        wherefound = np.where(found > 0)
        where_notfound = np.where(found==0)
        x_wherefound = x[wherefound]
        x_where_notfound = x[where_notfound]
        n_found = len(wherefound[0])
        n_notfound = len(where_notfound[0])

        # The number of failed 'attempts' at locations where the species was not found.
        # This is used in the likelihood by the CMVNSStepper object.
        n_neg = (found+others_found+zero)[where_notfound]
        
        # Evaluate the environmental surfaces at the data locations.
        if len(env_variables)>0:
            env_x = np.array([extract_environment(n, x * 180./np.pi) for n in env_variables]).T
        else:
            env_x = np.empty((len(x),0))
        env_x_wherefound = env_x[wherefound]    
        env_x_where_notfound = env_x[where_notfound]
        
        full_x_wherefound = np.hstack((x_wherefound, env_x_wherefound))
        full_x_wherefound_n = normalize_env(full_x_wherefound, env_means, env_stds)
        full_x_where_notfound = np.hstack((x_where_notfound, env_x_where_notfound))
        full_x_where_notfound_n = normalize_env(full_x_where_notfound, env_means, env_stds)
        
        p_eval_where_notfound = pm.Lambda('p_eval', 
            lambda p=p, C=C: p(full_x_where_notfound, offdiag=C(full_x_where_notfound_n, full_x_fr_n)), trace=False, 
                doc="The probability being within the range, evaluated on all the data locations where the species was not found.")

        # FIXME: This should be used by the CMVNSStepper and the constraint, if possible.
        # Might be too hard though.
        f_eval_wherefound = pm.Lambda('f_eval_wherefound', 
            lambda p=p, C=C: p(full_x_wherefound, f2p=identity, offdiag=C(full_x_wherefound_n, full_x_fr_n)), trace=False,
                doc="The suitability function evaluated everywhere the species was found.")
        
        from IPython.Debugger import Pdb
        Pdb(color_scheme='LightBG').set_trace() 
        # Enforce presence at the data locations with a constraint.
        constraint_dict = {'data': 
            Constraint(penalty_value = -1e100, logp=lambda f: -np.sum(f*(f<0)), 
                doc="A constraint enforcing presence at the data locations", 
                name='data_constraint', parents={'f':f_eval_wherefound})}
    
    if with_eo:
        
        # ====================================================================
        # = FIXME: The expert opinions are not supported by CMVNStepper yet. =
        # ====================================================================
        
        # ==============================
        # = Expert-opinion likelihoods =
        # ==============================
        
        p_eval_in = pm.Lambda('p_eval_in', 
            lambda p=p, C=C: p(full_x_in, offdiag=C(full_x_in_n, full_x_fr_n)), trace=False,
                doc="The probability of being within the range evaluated at the inducing points inside the EO region.")

        p_eval_out = pm.Lambda('p_eval_out', 
            lambda p=p, C=C: p(full_x_out, offdiag=C(full_x_out_n, full_x_fr_n)), trace=False,
                doc="The probability of being within the range evaluated at the inducing points outside the EO region")   

        p_eval_eo = pm.Lambda('p_eval_eo', 
            lambda p_eval_in=p_eval_in, p_eval_out=p_eval_out: np.concatenate((p_eval_in,p_eval_out)), trace=False,
                doc="The probability of being within the range evaluated at all the inducing points.")
    
        @pm.potential
        def not_all_present(p=p_eval_eo):
            """Potential that guards against global presence"""
            if np.all(p):
                return -np.inf
            else:
                return 0
        
        in_prob = pm.Lambda('in_prob', lambda p = p_eval_in: np.mean(p)*.9999+.00005,
            doc = "The probability that a uniformly-distributed point in the EO region is within the range")
        out_prob = pm.Lambda('out_prob', lambda p = p_eval_out: np.mean(p)*.9999+.00005,
            doc = "The probability that a uniformly-distributed point outside the EO region is within the range")

        alpha_out = pm.Uniform('alpha_out',0,1)
        beta_out = pm.Uniform('beta_out',1,10)            
        @pm.potential
        def out_factor(a=alpha_out, b=beta_out, p=out_prob):
            """The 'observation' that the range does not extend outside the EO region"""
            return pm.beta_like(p,a,b)
        eo_mean_out = pm.Lambda('eo_mean_out', 
            lambda a=alpha_out, b=beta_out: a/(a+b),
            doc="The 'likelihood mean' of the proportion of the non-EO region that is within the range")

        alpha_in = pm.Uniform('alpha_in',1,10)
        beta_in = pm.Uniform('beta_in',0,1)        
        @pm.potential
        def in_factor(a=alpha_in, b=beta_in, p=in_prob):
            """The 'observation' that the range fills the entire EO region"""
            return pm.beta_like(p,a,b)
        eo_mean_in = pm.Lambda('eo_mean_in', 
            lambda a=alpha_in, b=beta_in: a/(a+b),
            doc="The 'likelihood mean' of the proportion of the EO region that is outside the range")

        # # ========================
        # # = Hard expert opinions =
        # # ========================
        #
        # # FIXME: You need to reinstate f_eval_in, f_eval_out, f_eval_eo to use these.
        #     
        # env_eo = np.vstack((env_in, env_out))
        # pts_eo = np.vstack((pts_in,pts_out))
        # env_dict = dict(zip(env_variables,env_x.T))
        # env_dict_eo = dict(zip(env_variables,env_eo.T))
        # 
        # f_wherefound = np.ones(len(wherefound[0]))
        # f_in_eo = np.ones(n_in)
        # for k in constraint_fns.iterkeys():
        #     if k=='location':
        #         x_constraint = x
        #         x_constraint_eo = pts_eo
        #     else:
        #         x_constraint = env_dict[k]
        #         x_constraint_eo = env_dict_eo[k]
        #     
        #     # Make mighty sure that the constraint is satisfied at all datapoints.
        #     constraint_wherefound = constraint_fns[k](x=x_constraint[wherefound], f=f_wherefound)
        #     if constraint_wherefound>0:
        #         raise ValueError, 'Constraint function %s with input variable %s is violated the following locations where %s was found: \n%s' %\
        #                             (constraint_fns[k].__name__, k, species[1], x_wherefound[np.where(constraint_wherefound>0)]*180./np.pi)
        #                             
        #     # Make kind of sure that the constraint is satisfied in most of the expert opinion region.
        #     constraint_in_eo = constraint_fns[k](x=x_constraint_eo[:n_in], f=f_in_eo)
        #     if constraint_in_eo>0:
        #         warnings.warn('Constraint function %s with input variable %s is violated on %f of the expert opinion region for %s.' %\
        #                             (constraint_fns[k].__name__, k, constraint_in_eo/float(n_in), species[1]))
        #     
        #     # Make sure the constraint doesn't get violated in the range of the species.
        #     constraint_dict[k] = Constraint(penalty_value = -1e100, logp=constraint_fns[k], 
        #         doc="", 
        #         name='%s_constraint'%k, parents={'x': x_constraint_eo, 'f': f_eval_eo})

    out = locals()
    out.update(spatial_variables)
    return out

def species_stepmethods(M, interval=None, sleep_interval=1):
    """
    Adds appropriate step methods to M.
    """
    bases = filter(lambda x: isinstance(x, OrthogonalBasis), M.stochastics)
    nonbases = set(filter(lambda x: True-isinstance(x, OrthogonalBasis), M.stochastics))

    for alone in [M.f_fr, M.alpha_in, M.alpha_out, M.beta_in, M.beta_out, M.p_find]:
        nonbases.discard(alone)
            
    # Standard step methods
    # M.use_step_method(pm.AdaptiveMetropolis, M.vals + list(M.val.extended_parents), delay=2000)
    # for p in M.val.extended_parents:
    #     M.use_step_method(pm.Metropolis, p)
    # M.use_step_method(pm.AdaptiveMetropolis, M.vals)
    # [M.use_step_method(pm.Metropolis,v) for v in M.vals]
    
    # FIXME: CMVNLStepper is not taking into account the EO or any of the hard constraints right now.
    if interval is None:
        M.use_step_method(CMVNLStepper, M.f_fr, -M.od_wherefound, np.zeros(len(M.x_wherefound)), M.od_where_notfound, M.n_neg, M.p_find, pri_S=M.L_fr, pri_M=None, n_cycles=100, pri_S_type='tri')
    else:
        for i in xrange(0,len(M.f_fr.value),interval):
            M.use_step_method(SubsetMetropolis, M.f_fr, i, interval, sleep_interval)
    
    # Weird step methods
    # M.use_step_method(RayMetropolis, M.vals, 1)
    
    # Givens step method
    for b in bases:
        M.use_step_method(GivensStepper, b)    

def restore_species_MCMC(session, dbpath):
    db = pm.database.hdf5.load(dbpath)
    metadata = db._h5file.root.metadata[0]
    model = make_model(session, **metadata)        
    M=LatchingMCMC(model, db=db)
    species_stepmethods(M, interval=5)
    M.restore_sampler_state
    M.restore_sm_state
    M.__dict__.update(metadata)
    return M

def add_metadata(hf, kwds, species, spatial_submodel):
    hf.createVLArray('/','metadata', atom=tb.ObjectAtom())
    metadata = {}
    metadata.update(kwds)
    metadata['species']=species
    metadata['spatial_submodel']=spatial_submodel    
    hf.root.metadata.append(metadata)

def species_MCMC(session, species, spatial_submodel, **kwds):
    print 'Environment variables: ',kwds['env_variables']
    print 'Constraints: ',kwds['constraint_fns']
    print 'Spatial submodel: ',spatial_submodel.__name__
    print 'Species: ',species[1]

    # ====================================
    # = First stage: Satisfy constraints =
    # ====================================
    model = make_model(session, species, spatial_submodel, **kwds)        
    M1=LatchingMCMC(model, db='ram')
    species_stepmethods(M1, interval=5, sleep_interval=20)
    print 'Attempting to satisfy constraints'
    M1.isample(1,burn=1000)
    print 'Done!'
    
    # =======================================
    # = Second stage: sample from posterior =
    # =======================================
    M2=pm.MCMC(model, db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')

    # Create data object. Don't create it far the first stage, because all you want to do at that stage is find a legal initial value.
    M2.data = pm.Binomial('data', n=M2.n_neg, p=M2.p_eval_where_notfound*M2.p_find, value=M2.n_notfound, observed=True, trace=False)            
    species_stepmethods(M2, interval=10, sleep_interval=20)
    add_metadata(M.db._h5file, kwds, species, spatial_submodel)
    
    # Try to initialize full-rank field to a reasonable value.
    for i in xrange(10):
        M2.step_method_dict[M2.f_fr][0].step()

    # Make sure data_constraint is evaluated before data likelihood, to avoid as meany heavy computations as possible.
    M2.assign_step_methods()
    for sm in M2.step_methods:
        for i in xrange(len(sm.markov_blanket)):
            if sm.markov_blanket[i] is M2.data:
                sm.markov_blanket.append(M2.data)
                sm.markov_blanket.pop(i)
    
    return M2