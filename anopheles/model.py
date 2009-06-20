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
import cov_prior
from mahalanobis_covariance import mahalanobis_covariance
from map_utils import multipoly_sample


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

def ghetto_spatial_submodel(**kerap):
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
    
    @pm.deterministic(trace=False)
    def f(val = bump_eigenvalues, vec = bump_eigenvectors, ctr = ctr, amp=amp):
        "A stupid hill, using Euclidean distance."
        def f(x, val=val, vec=vec, ctr=ctr):
            dev = x-ctr
            tdev = np.dot(dev, vec)
            if len(dev.shape)==1:
                ax=0
            else:
                ax=1
            return pm.invlogit(np.sum(tdev**2/val,axis=ax)*amp)
        return f
        
    return locals()

def multipoint_to_ndarray(mp):
    "Converts a multipont to a coordinate array IN RADIANS."
    return np.array([[p.x, p.y] for p in mp.geoms])*np.pi/180.
    
def make_model(session, species, spatial_submodel = ghetto_spatial_submodel):

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
    # FIXME: Make sure you're interpreting the queries correctly!
    x = []
    breaks = [0]
    found = []
    
    for site in sites:
        x.append(multipoint_to_ndarray(site[0]))
        breaks.append(breaks[-1] + len(site[0].geoms))
        found.append(site[1] is not None)

    breaks = np.array(breaks)
    x = np.concatenate(x)
    found = np.array(found)

    # TODO: Evaluate f all at once, then slice the output. Should help a lot.
    # TODO: Also vectorize the data.
    f_eval = f(x)


    @pm.deterministic(trace=False)
    def p_find_somewhere(f_eval=f_eval, p_find=p_find, breaks=breaks):
        out = np.empty(len(breaks)-1)
        for i in xrange(len(breaks)-1):
            fe = f_eval[breaks[i]:breaks[i+1]]
            out[i]=1.-np.prod(fe*(1-p_find) + 1.-fe)
        return out
    
    data =pm.Bernoulli('points', p_find_somewhere, value=found, observed=True)
            
    
    # ==============================
    # = Expert-opinion likelihoods =
    # ==============================
    
    out = locals()
    out.update(spatial_submodel)
    return out
    
if __name__ == '__main__':
    session = Session()
    species = list_species(session)    
    M=pm.MCMC(make_model(session, species[1]))