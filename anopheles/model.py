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

__all__ = ['make_model', 'species_MCMC', 'threshold','invlogit', 'restore_species_MCMC','identity','threshold','invlogit']

def identity(x):
    return x

def threshold(f):
    return f > 0
    
def invlogit(f):
    return pm.flib.invlogit(f.ravel()).reshape(f.shape)

def evaluation_group(C,U,p,x,x_p,f2p,suffix,doc=''):
    od = pm.Lambda('od_%s'%suffix, lambda C=C, U=U: compute_offdiag(C,U,x,x_p), trace=False)
    f_eval = pm.Lambda('f_eval_%s'%suffix, lambda p=p, od=od: p(x_p, f2p=identity, offdiag=od), trace=False, doc=doc)
    p_eval = pm.Lambda('p_eval_%s'%suffix, lambda f=f_eval, f2p=f2p: f2p(f), trace=False)
    return od, f_eval, p_eval
    
def make_model(session, species, spatial_submodel, with_eo = True, with_data = True, env_variables = (), constraint_fns={}, n_inducing=1000, f2p=threshold):
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
    
    pts_in, pts_out = sample_eo(session, species, n_inducing)
    
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
    
    p_find = pm.Uniform('p_find',0,1,value=.99,doc="Probability of finding a species within its range.",observed=True)

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
    x_fr = x_eo#[::2]
    full_x_fr = full_x_eo#[::2]
    
    full_x_fr_n = normalize_env(full_x_fr, env_means, env_stds)

    # ============================
    # = Call to spatial submodel =
    # ============================
    spatial_variables = spatial_submodel(**locals())
    p = spatial_variables['p']
    f_fr = spatial_variables['f_fr']
    if spatial_variables.has_key('val'):
        val = spatial_variables['val']
    vec = spatial_variables['vec']
    C = spatial_variables['C']
    U_fr = spatial_variables['U_fr']
    g_fr = spatial_variables['g_fr']
    
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
        n_pos = (found+others_found+zero)[wherefound]
        
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
        
        od_where_notfound, f_eval_where_notfound, p_eval_where_notfound = \
            evaluation_group(C,U_fr,p,full_x_fr_n,full_x_where_notfound_n,f2p,'where_notfound',
                doc="The suitability function evaluated on all the data locations where the species was not found.")

        od_wherefound, f_eval_wherefound, p_eval_wherefound = \
            evaluation_group(C,U_fr,p,full_x_fr_n,full_x_wherefound_n,f2p,'wherefound',
                doc="The suitability function evaluated everywhere the species was found.")
                
        p_eval_wheredata = pm.Lambda('p_eval_wheredata', lambda p1=p_eval_wherefound, p2=p_eval_where_notfound: np.hstack((p1,p2)), trace=False)

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
        
        od_in, f_eval_in, p_eval_in = \
            evaluation_group(C,U_fr,p,full_x_fr_n,full_x_in_n,f2p,'in',
                doc="The suitability function evaluated at the inducing points inside the EO region.")

        od_out, f_eval_out, p_eval_out = \
            evaluation_group(C,U_fr,p,full_x_fr_n,full_x_out_n,f2p,'out',
                doc="The suitability function evaluated at the inducing points outside the EO region.")

        p_eval_eo = pm.Lambda('p_eval_eo', 
            lambda p_eval_in=p_eval_in, p_eval_out=p_eval_out: np.concatenate((p_eval_in,p_eval_out)), trace=False,
                doc="The probability of being within the range evaluated at all the inducing points.")
    
        @pm.potential
        def not_all_present(p=p_eval_eo):
            """Potential that guards against global presence"""
            if np.all(p):
                return -1.e100
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
    scalar_nonbases = filter(lambda x: np.prod(np.shape(x.value))<=1, nonbases)

    for s in scalar_nonbases:
        M.use_step_method(pm.Metropolis, s)
        if s in M.f_fr.extended_parents:
            M.step_method_dict[s][0].adaptive_scale_factor=.01
        
    for s in nonbases - set(scalar_nonbases):
        if s is not M.f_fr:
            M.use_step_method(pm.AdaptiveMetropolis, s, scales={s: .0001*np.ones(np.shape(s.value))})

    M.use_step_method(pm.AdaptiveMetropolis, scalar_nonbases)
    # if hasattr(M, 'val'):
    #     if not isinstance(M.val, pm.Stochastic):
    #         M.use_step_method(pm.AdaptiveMetropolis, M.val)    
            
    # FIXME: CMVNLStepper is not taking into account the EO or any of the hard constraints right now.
    if interval is None:
        # pm.gp.trisolve(U_fr,pm.gp.trisolve(U_fr,f_fr,uplo='U',transa='T'),uplo='U',transa='N',inplace=True)
        # @pm.deterministic(trace=False)
        # def B(od=M.od_wherefound, U=M.U_fr):
        #     o1 = pm.gp.trisolve(U,(-od.T).copy('F'),uplo='U',transa='T',inplace=True)
        #     o2 = pm.gp.trisolve(U,o1,uplo='U',transa='N',inplace=True)
        #     return np.asarray(o2.T,order='F')
        #     
        # @pm.deterministic(trace=False)
        # def Bl(od=M.od_where_notfound, U=M.U_fr):
        #     o1 = pm.gp.trisolve(U,(od.T).copy('F'),uplo='U',transa='T',inplace=True)
        #     o2 = pm.gp.trisolve(U,o1,uplo='U',transa='N',inplace=True)
        #     return np.asarray(o2.T,order='F')
        
        likelihood_offdiags = [M.od_where_notfound, M.od_in, M.od_out]
        constraint_offdiags = [M.od_wherefound]
        # 1 = must be above 0, -1 = must be below 0.
        constraint_signs = [1]
        M.use_step_method(CMVNLStepper, M.f_fr, M.g_fr, M.U_fr, likelihood_offdiags, constraint_offdiags, constraint_signs)
        
        # M.use_step_method(pm.NoStepper, M.f_fr)
        # M.sm_ = CMVNLStepper(M.f_fr, B, np.zeros(len(M.x_wherefound)), Bl, M.n_neg, M.p_find, pri_S=M.L_fr, pri_M=None, n_cycles=100, pri_S_type='tri')
        # M.use_step_method(CMVNLStepper, M.f_fr, B, np.zeros(len(M.x_wherefound)), Bl, M.n_neg, M.p_find, pri_S=M.L_fr, pri_M=None, n_cycles=100, pri_S_type='tri')
        # M.use_step_method(pm.AdaptiveMetropolis, M.f_fr)
        # M.use_step_method(pm.AdaptiveMetropolis, M.f_fr, scales={M.f_fr: M.f_fr.value*0+.0001})
    else:
        for i in xrange(0,len(M.f_fr.value),interval):
            M.use_step_method(SubsetMetropolis, M.f_fr, i, interval, sleep_interval)
    
    # Weird step methods
    # M.use_step_method(RayMetropolis, M.vals, 1)
    
    # Givens step method
    for b in bases:
        M.use_step_method(GivensStepper, b)    
        M.step_method_dict[b][0].adaptive_scale_factor=.1

def add_data(M2):
    M2.data = pm.Binomial('data', 
        n=np.hstack((M2.n_pos, M2.n_neg)), 
        p=M2.p_eval_wheredata*M2.p_find, 
        value=np.hstack((M2.found[M2.wherefound], np.zeros(M2.n_notfound))),
        observed=True, trace=False)
    M2.observed_stochastics.add(M2.data)
    M2.variables.add(M2.data)        
    M2.nodes.add(M2.data)


def restore_species_MCMC(session, dbpath):

    # Load the database from the disk
    db = pm.database.hdf5.load(dbpath)
    
    # Recover MCMC object, restore variables' states, add data
    metadata = db._h5file.root.metadata[0]
    model = make_model(session, **metadata)
    M=pm.MCMC(model, db=db)
    M.restore_sampler_state()
    for c in filter(lambda x: isinstance(x,Constraint), M.potentials):
        c.close()
    add_data(M)
    
    
    # Assign step methods and restore states
    species_stepmethods(M)
    M.assign_step_methods()
    for sm in M.step_methods:
        for i in xrange(len(sm.markov_blanket)):
            if sm.markov_blanket[i] is M.data:
                sm.markov_blanket.append(M.data)
                sm.markov_blanket.pop(i)
    M.restore_sm_state()
    
    # Add the information about the species and the spatial submodel to M.
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
    M1=LatchingMCMC(make_model(session, species, spatial_submodel, **kwds), db='ram')
    new_val = M1.f_fr.value*0+.1
    # new_val[len(M1.pts_in):] = .001
    M1.f_fr.value = new_val
    
    # species_stepmethods(M1, interval=5, sleep_interval=20)
    species_stepmethods(M1)
    for s in M1.stochastics:
        for sm in M1.step_method_dict[s]:
            sm.step()
            
    print 'Attempting to satisfy constraints'
    M1.isample(1)
    print 'Done!'
    
    # =======================================
    # = Second stage: sample from posterior =
    # =======================================
    M2=pm.MCMC(make_model(session, species, spatial_submodel, **kwds), db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    for s2 in M2.stochastics:
        for s1 in M1.stochastics:
            if s2.__name__ == s1.__name__:
                s2.value = s1.value
    for c in filter(lambda x: isinstance(x,Constraint), M2.potentials):
        c.close()

    # Create data object. Don't create it far the first stage, because all you want to do at that stage is find a legal initial value.
    add_data(M2)
    species_stepmethods(M2)
    
    add_metadata(M2.db._h5file, kwds, species, spatial_submodel)
    
    # Try to initialize full-rank field to a reasonable value.
    # for i in xrange(10):
    #     M2.step_method_dict[M2.f_fr][0].step()

    # Make sure data_constraint is evaluated before data likelihood, to avoid as meany heavy computations as possible.
    M2.assign_step_methods()
    for sm in M2.step_methods:
        for i in xrange(len(sm.markov_blanket)):
            if sm.markov_blanket[i] is M2.data:
                sm.markov_blanket.append(M2.data)
                sm.markov_blanket.pop(i)
    
    return M2