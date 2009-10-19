# import matplotlib
# matplotlib.use('pdf')
import anopheles
from anopheles_query import Session
from cov_prior import OrthogonalBasis, GivensStepper
from pymc import AdaptiveMetropolis
import pymc as pm

from anopheles.species.darlingi import *
# from anopheles.species.gambiae import *
# from anopheles.species.arabiensis import *
# from anopheles.species.stephensi import *

s = Session()
species = dict([sp[::-1] for sp in anopheles.list_species(s)])
species_tup = (species[species_name], species_name)

from mpl_toolkits import basemap
import pylab as pl

pl.close('all')

mask, x, img_extent = anopheles.make_covering_raster(100, env)

spatial_submodel = anopheles.nogp_spatial_env
n_in = n_out = 1000

# spatial_submodel = lr_spatial_env
# n_in = n_out = 1000

# spatial_submodel = spatial_env
# n_out = 400
# n_in = 100

M = anopheles.species_MCMC(s, species_tup, spatial_submodel, with_eo = True, with_data = True, env_variables = env, constraint_fns=cf,n_in=n_in,n_out=n_out)

# M.assign_step_methods()
# sf=M.step_method_dict[M.f_fr][0]    
# ss=M.step_method_dict[M.p_find][0]
# sa = M.step_method_dict[M.ctr][0]

# M.assign_step_methods()
M.isample(50000,0,10,verbose=0)
# pm.Matplot.plot(M)
# for name in ['ctr','val','coefs','const']:
#     pl.figure()
#     pl.plot(M.trace(name)[:])
#     pl.title(name)
# 
# # mask, x, img_extent = make_covering_raster(2)
# # b = basemap.Basemap(*img_extent)
# # out = M.p.value(x)
# # arr = np.ma.masked_array(out, mask=True-mask)
# # b.imshow(arr.T, interpolation='nearest')
# # pl.colorbar()
# pl.figure()
# anopheles.current_state_map(M, s, species[species_num], mask, x, img_extent, thin=100)
# pl.title('Final')
# pl.savefig('final.pdf')
# pl.figure()
# pl.plot(M.trace('out_prob')[:],'b-',label='out')
# pl.plot(M.trace('in_prob')[:],'r-',label='in')    
# pl.legend(loc=0)
# 
# pl.figure()
out, arr = anopheles.presence_map(M, s, species_tup, thin=100, burn=500, trace_thin=1)
# pl.figure()
# x_disp, samps = mean_response_samples(M, -1, 10, burn=100, thin=1)
# for s in samps:
#     pl.plot(x_disp, s)
# pl.savefig('prob_%s.pdf'%species_name)
# 
# pl.figure()
# p_atfound = probability_traces(M)
# p_atnotfound = probability_traces(M,False)
# pl.savefig('presence.pdf')