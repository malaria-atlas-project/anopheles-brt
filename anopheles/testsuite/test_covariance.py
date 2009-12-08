import anopheles
from anopheles_query import Session
from cov_prior import OrthogonalBasis, GivensStepper
from pymc import AdaptiveMetropolis
import pymc as pm

# from anopheles.species.darlingi import *
from anopheles.species.gambiae import *
# from anopheles.species.arabiensis import *
# from anopheles.species.stephensi import *

s = Session()
species = dict([sp[::-1] for sp in anopheles.list_species(s)])
species_tup = (species[species_name], species_name)

from mpl_toolkits import basemap
import pylab as pl

pl.close('all')

mask, x, img_extent = anopheles.make_covering_raster(100, env)
mask, x, img_extent = anopheles.make_covering_raster(20, env)
# outside_lat = (x[:,1]*180./np.pi>38)+(x[:,1]*180./np.pi<-36)
# outside_lon = (x[:,0]*180./np.pi>56)+(x[:,0]*180./np.pi<-18)
mask, x, img_extent = anopheles.subset_x(mask,x,img_extent,(-18,-36,56,38))

spatial_submodel = anopheles.lr_spatial_env
# spatial_submodel = anopheles.nogp_spatial_env
# n_in = n_out = 2

# spatial_submodel = lr_spatial_env
n_in = n_out = 1000

# spatial_submodel = spatial_env
# n_out = 400
# n_in = 100

M = pm.MCMC(anopheles.model.make_model(s, species_tup, spatial_submodel, with_eo = True, with_data = True, env_variables = env, constraint_fns=cf,n_in=n_in,n_out=n_out))

pl.close('all')
M.spatial_variables['fracs'].value = np.array([.001,.998])
M.f_fr.rand()
anopheles.current_state_map(M, s, species_tup, mask, x, img_extent, thin=100, f2p=anopheles.model.identity)
pl.colorbar()
pl.title('All spatial')

pl.figure()
M.spatial_variables['fracs'].value = np.array([.001,.001])
M.f_fr.rand()
anopheles.current_state_map(M, s, species_tup, mask, x, img_extent, thin=100, f2p=anopheles.model.identity)
pl.colorbar()
pl.title('All environmental')

pl.figure()
M.spatial_variables['fracs'].value = np.array([.998,.001])
M.f_fr.rand()
anopheles.current_state_map(M, s, species_tup, mask, x, img_extent, thin=100, f2p=anopheles.model.identity)
pl.colorbar()
pl.title('Mostly constant')

# if __name__ == '__main__':
#     nose.runmodule()
