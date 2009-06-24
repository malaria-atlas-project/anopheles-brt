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
from map_utils import multipoly_sample
from spatial_submodels import *
from utils import bin_ubl
import datetime

__all__ = ['make_model', 'species_MCMC']
    
def make_model(session, species, spatial_submodel):

    # =========
    # = Query =
    # =========
    sites, eo = species_query(session, species[0])
    
    
    # ==========
    # = Priors =
    # ==========
    p_find = pm.Uniform('p_find',0,1)
    spatial_submodel = spatial_submodel(**locals())
    f = spatial_submodel['f']
    
    # Forget about non-records
    sites = filter(lambda s:s[0] is not None, sites)
    
    
    # ===============
    # = Likelihoods =
    # ===============
    
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

    @observed
    @pm.stochastic
    def points(value = [found, others_found, zero], f_eval=f_eval, p_find=p_find, breaks=breaks):
        return bin_ubls(value[0], value[0]+value[1]+value[2], p_find, breaks, f_eval)
    
    # ==============================
    # = Expert-opinion likelihoods =
    # ==============================
    
    out = locals()
    out.update(spatial_submodel)
    return out

def species_MCMC(session, species, spatial_submodel, db=None):
    if db is None:
        M=pm.MCMC(make_model(session, species, spatial_hill), db='hdf5', complevel=1, dbname=species[1]+str(datetime.datetime.now())+'.hdf5')
    else:
        M=pm.MCMC(make_model(session, species, spatial_hill), db=db)
    return M
    
if __name__ == '__main__':
    