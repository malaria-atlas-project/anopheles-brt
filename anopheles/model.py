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
from mahalanobis_covariance import mahalanobis_covariance
from map_utils import multipoly_sample, grid_convert
from spatial_submodels import *
from utils import bin_ubls
import datetime

__all__ = ['make_model', 'species_MCMC', 'probability_traces', 'probability_map']
    
def make_model(session, species, spatial_submodel, with_eo = True):
    """
    Generates a PyMC probability model with a plug-in spatial submodel.
    The likelihood and expert-opinion layers are common.
    """

    # =========
    # = Query =
    # =========
    
    sites, eo = species_query(session, species[0])
    
    
    # ==========
    # = Priors =
    # ==========
    
    p_find = pm.Uniform('p_find',0,1)
    spatial_variables = spatial_submodel(**locals())
    f = spatial_variables['f']
    
    # Forget about non-records
    sites = filter(lambda s:s[0] is not None, sites)
    
    
    # ==============
    # = Likelihood =
    # ==============
    
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

    f_eval = f(x)

    @pm.observed
    @pm.stochastic
    def points(value = [found, others_found, zero], f_eval=f_eval, p_find=p_find, breaks=breaks):
        return bin_ubls(value[0], value[0]+value[1]+value[2], p_find, breaks, f_eval)
    
    
    # ==============================
    # = Expert-opinion likelihoods =
    # ==============================

    if with_eo:
        pts_in, pts_out = sample_eo(session, species, 1000, 1000)
        sens_strength = pm.Uninformative('sens_strength',1000,observed=True)
        spec_strength = pm.Uninformative('spec_strength',1000,observed=True)    
        in_prob = pm.Lambda('in_prob', lambda f=f, x=pts_in: np.mean(f(x)))
        out_prob = pm.Lambda('out_prob', lambda f=f, x=pts_out: np.mean(f(x)))    
    
        @pm.potential
        def spec_factor(s=spec_strength, p=out_prob):
            return s*p
        
        @pm.potential
        def sens_factor(s=sens_strength, p=out_prob):
            return s*(1.-p)
    
    
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
        f = M.trace('f')[i:i+1][0]
        vals.append(f(x))
    return np.array(vals)
    
def presence_map(M, session, species, burn=0, worldwide=True, thin=1, **kwds):
    "Converts the trace to a map of presence probability."
    
    # FIXME: Use the proper land-sea mask, served from the db, to do this.
    import mbgw
    from mpl_toolkits import basemap
    import pylab as pl
    
    a=getattr(mbgw.auxiliary_data,'landSea-e')
    
    lon = a.long[::thin]
    lat = a.lat[::thin][::-1]
    # mask = a.data[::thin,::thin]
    mask = grid_convert(a.data[::thin,::thin],'y-x+','x+y-')
    
    extent = [-180,-90,180,90]
    
    if not worldwide:
        extent = map_extents(pos_recs, eo)
        where_inlon = np.where((lon>=extent[0]) * (lon <= extent[2]))
        where_inlat = np.where((lon>=extent[0]) * (lon <= extent[2]))        
        lon = lon[where_inlon]
        lat = lat[where_inlat]
        mask = mask[where_inlon,:][:,where_inlat]
        
    img_extent = [lon.min(), lat.min(), lon.max(), lat.max()]
        
    lat_grid, lon_grid = np.meshgrid(lat*np.pi/180.,lon*np.pi/180.)
    x=np.dstack((lon_grid,lat_grid))
    out = np.zeros(mask.shape)

    for i in xrange(M._cur_trace_index):
        f = M.trace('f')[i:i+1][0]
        out += f(x)/M._cur_trace_index
    
    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(arr.T, interpolation='nearest')    
    pl.colorbar()    
    plot_species(session, species[0], species[1], b, negs=True, **kwds)    

def species_MCMC(session, species, spatial_submodel, db=None):
    if db is None:
        M=pm.MCMC(make_model(session, species, spatial_hill), db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    else:
        M=pm.MCMC(make_model(session, species, spatial_hill), db=db)
    return M
    
if __name__ == '__main__':
    s = Session()
    species = list_species(s)
    m=make_model(s, species[1], spatial_hill)
    M = pm.MCMC(m)
    M.isample(10000,0,10)
    
    presence_map(M, s, species[1], thin=2, burn=300)

    p_atfound = probability_traces(M)
    p_atnotfound = probability_traces(M,False)
    