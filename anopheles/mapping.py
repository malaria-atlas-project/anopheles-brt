import numpy as np
from env_data import *
from map_utils import reconcile_multiple_rasters, grid_convert
from query_to_rec import *

__all__ = ['make_covering_raster','presence_map','current_state_map','current_state_slice']

# TODO: reconcile_multiple_rasters and this function need a maxres argument.
def make_covering_raster(thin=1, env_variables=(), **kwds):
    # FIXME: This needs to be tolerant of multiple rasters in the environmental variables and land/sea mask.
    import mbgw
    from mbgw import auxiliary_data

    a=getattr(mbgw.auxiliary_data,'landSea-e')
    
    lon,lat,layers = reconcile_multiple_rasters([a]+[get_datafile(n) for n in env_variables])

    lon = lon[::thin]
    lat = lat[::thin][::-1]
    mask = grid_convert(layers[0][::thin,::thin],'y-x+','x+y-')
    mask = np.round(mask).astype(bool)

    img_extent = [lon.min(), lat.min(), lon.max(), lat.max()]

    lat_grid, lon_grid = np.meshgrid(lat*np.pi/180.,lon*np.pi/180.)
    x=np.dstack([lon_grid,lat_grid]+[grid_convert(l[::thin,::thin],'y-x+','x+y-') for l in layers[1:]])

    return mask, x, img_extent

def presence_map(M, session, species, burn=0, thin=1, trace_thin=1, **kwds):
    "Converts the trace to a map of presence probability."
    
    from mpl_toolkits import basemap
    import pylab as pl
    
    mask, x, img_extent = make_covering_raster(thin, M.env_variables, **kwds)

    out = np.zeros(mask.shape)

    for i in xrange(burn, M._cur_trace_index, trace_thin):
        p = M.trace('p')[:][i]
        pe = p(x)
        out += pe/float(M._cur_trace_index-burn)
    
    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(arr.T, interpolation='nearest')    
    pl.colorbar()    
    plot_species(session, species[0], species[1], b, negs=True, **kwds)    
    return out, arr
    
def current_state_map(M, session, species, mask, x, img_extent, thin=1, **kwds):
    "Maps the current state of the model."
    
    from mpl_toolkits import basemap
    import pylab as pl
    import time

    out = M.p.value(x)

    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(arr.T, interpolation='nearest')    
    b.plot(M.x_fr.value[:,0]*180./np.pi, M.x_fr.value[:,1]*180./np.pi, 'g.', markersize=2)
    pl.colorbar()    

    return out, arr
    
def current_state_slice(M, axis):
    "Plots the current state of the model along a slice."
    import pylab as pl
    
    N = 100

    x = M.x_eo[:,axis]
    x = np.linspace(x.min(), x.max(),N)
    
    other_x = np.array([np.mean(M.x_eo,axis=0)]*N)
    
    other_x[:,axis] = x
    
    out = M.p.value(other_x)
    pl.plot(x, out)
    
    return other_x, out