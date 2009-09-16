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

import pymc as pm
import numpy as np
from models import Session
from query_to_rec import *
from env_data import *
from mahalanobis_covariance import mahalanobis_covariance
from map_utils import multipoly_sample
from mapping import *
from spatial_submodels import *
from utils import bin_ubls
import datetime

__all__ = ['make_model', 'species_MCMC', 'probability_traces','potential_traces']
    
def make_model(session, species, spatial_submodel, with_eo = True, with_data = True, env_variables = ()):
    """
    Generates a PyMC probability model with a plug-in spatial submodel.
    The likelihood and expert-opinion layers are common.
    """

    # =========
    # = Query =
    # =========
    
    sites, eo = species_query(session, species[0])
    pts_in, pts_out = sample_eo(session, species, 1000, 1000)
    
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
    spatial_variables = spatial_submodel(**locals())
    p = spatial_variables['p']
    
    # Forget about non-records
    sites = filter(lambda s:s[0] is not None, sites)
    
    # ==============
    # = Likelihood =
    # ==============

    if with_data:
        x = []
        breaks = [0]
        found = []
        zero = []
        others_found = []
        totals = []
    
        for site in sites:
            x.append(multipoint_to_ndarray(site[0]))
            breaks.append(breaks[-1] + len(site[0].geoms))
            found.append(site[1] if site[1] is not None else 0)
            zero.append(site[2] if site[2] is not None else 0)
            others_found.append(site[3] if site[3] is not None else 0)
            totals.append(site[4])

        breaks = np.array(breaks)
        x = np.concatenate(x)
        found = np.array(found)
        zero = np.array(zero)
        others_found = np.array(others_found)
        
        if len(env_variables)>0:
            env_x = np.array([extract_environment(n, x * 180./np.pi) for n in env_variables]).T
        else:
            env_x = np.empty((len(x),0))

        p_eval = p(np.hstack((x, env_x)))

        @pm.observed
        @pm.stochastic(trace=False)
        def data(value = [found, others_found, zero], p_eval=p_eval, p_find=p_find, breaks=breaks):
            return bin_ubls(value[0], value[0]+value[1]+value[2], p_find, breaks, p_eval)
    
    # ==============================
    # = Expert-opinion likelihoods =
    # ==============================
    
    if with_eo:
        # sens_strength = pm.Uninformative('sens_strength',1000,observed=True)
        # spec_strength = pm.Uninformative('spec_strength',1000,observed=True)    
        if spatial_variables.has_key('in_prob'):
            in_prob = spatial_variables['in_prob']
            out_prob = spatial_variables['out_prob']
        else:
            in_prob = pm.Lambda('in_prob', lambda p=p, x=pts_in, e=env_in: np.mean(p(np.hstack((x, e))))*.9999+.00005)
            out_prob = pm.Lambda('out_prob', lambda p=p, x=pts_out, e=env_out: np.mean(p(np.hstack((x,e))))*.9999+.00005)    

        # @pm.potential
        # def out_factor(p=out_prob):
        #     if p == 0 or p == 1:
        #         return -np.inf
        #     return pm.flib.logit(1.-p)*100
        # 
        # @pm.potential
        # def in_factor(p=in_prob):
        #     if p == 0 or p == 1:
        #         return -np.inf
        #     return pm.flib.logit(p)*100
            
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
    if db is None:
        M=pm.MCMC(make_model(session, species, spatial_submodel, **kwds), db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    else:
        M=pm.MCMC(make_model(session, species, spatial_submodel, **kwds), db=db)
    scalar_stochastics = filter(lambda s: np.prod(np.shape(s.value))<=1, M.stochastics)
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
if __name__ == '__main__':
    s = Session()
    species = list_species(s)
    import mbgw

    # m=make_model(s, species[1], spatial_hill, with_data=False)
    # m=make_model(s, species[1], lr_spatial)
    # M = pm.MCMC(m)
    
    from mpl_toolkits import basemap
    import pylab as pl
    species_num = 7
    
    # M = species_MCMC(s, species[species_num], lr_spatial, with_eo = True, with_data = True, env_variables = [])
    env = ['MODIS-hdf5/daytime-land-temp.mean.geographic.world.2001-to-2006',
            'MODIS-hdf5/evi.mean.geographic.world.2001-to-2006',
            'MODIS-hdf5/nighttime-land-temp.mean.geographic.world.2001-to-2006',
            'MODIS-hdf5/raw-data.elevation.geographic.world.version-5']
    # from map_utils import reconcile_multiple_rasters
    # o = reconcile_multiple_rasters([getattr(mbgw.auxiliarqy_data,'landSea-e')]+[get_datafile(n) for n in env])
    # import pylab as pl
    # for a in o[2]:
    #     pl.figure()
    #     pl.imshow(a)
    # env = ['public/anopheles/MARA','public/anopheles/SCI']
    M = species_MCMC(s, species[species_num], lr_spatial_env, with_eo = True, with_data = True, env_variables = env)
    
    pl.close('all')
    mask, x, img_extent = make_covering_raster(5, env)
    current_state_map(M, s, species[species_num], mask, x, img_extent, thin=1)
    pl.title('Initial')
    M.assign_step_methods()
    sf=M.step_method_dict[M.f_fr][0]
    ss=M.step_method_dict[M.p_find][0]
    
    for  i in xrange(100):
        try:
            M.data.logp
        except pm.ZeroProbability:
            M.f_fr.rand()
        
    M.isample(10000,0,10)
    
    # mask, x, img_extent = make_covering_raster(2)
    # b = basemap.Basemap(*img_extent)
    # out = M.p.value(x)
    # arr = np.ma.masked_array(out, mask=True-mask)
    # b.imshow(arr.T, interpolation='nearest')
    # pl.colorbar()
    pl.figure()
    current_state_map(M, s, species[species_num], mask, x, img_extent, thin=1)
    pl.title('Final')
    pl.figure()
    pl.plot(M.trace('out_prob')[:],'b-',label='out')
    pl.plot(M.trace('in_prob')[:],'r-',label='in')    
    pl.legend(loc=0)
    pl.figure()
    out, arr = presence_map(M, s, species[species_num], thin=5, burn=500, trace_thin=1)
    # pl.figure()
    # x_disp, samps = mean_response_samples(M, -1, 10, burn=100, thin=1)
    # for s in samps:
    #     pl.plot(x_disp, s)
    
    # p_atfound = probability_traces(M)
    # p_atnotfound = probability_traces(M,False)