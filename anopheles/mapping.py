import pylab as pl
import time
import numpy as np
from env_data import *
from map_utils import grid_convert
from query_to_rec import *
import hashlib
import tables
import os
import pymc as pm
from mpl_toolkits import basemap

__all__ = ['presence_map','current_state_map','current_state_slice']

try:
    from map_utils import reconcile_multiple_rasters
    def make_covering_raster(thin=1, env_variables=(), **kwds):
        

        a='MODIS-hdf5/raw-data.land-water.geographic.world.version-4'
        
        names = [a]+env_variables
        names.sort()
        cache_fname = 'rasters_%i_%s_.hdf5'%(thin, hashlib.sha1('~*~lala~*~oolala'.join(names)).hexdigest())
        
        cache_found = False
        if 'anopheles-caches' in os.listdir('.'):
            if cache_fname in os.listdir('anopheles-caches'):
                hf = tables.openFile(os.path.join('anopheles-caches',cache_fname))
                lon = hf.root.lon[:]
                lat = hf.root.lat[:]
                layers = []
                for n in [a] + env_variables:
                    layers.append(getattr(hf.root,os.path.split(n)[1])[:])
                hf.close()
                cache_found = True
        
        if not cache_found:
            lon,lat,layers = reconcile_multiple_rasters([get_datafile(n) for n in [a] + env_variables], thin=thin)
            hf = tables.openFile(os.path.join('anopheles-caches',cache_fname),'w')            
            hf.createArray('/','lon',lon)
            hf.createArray('/','lat',lat)
            names = [a] + env_variables
            for i in xrange(len(layers)):
                hf_arr=hf.createCArray('/',os.path.split(names[i])[1],shape=layers[i].shape,chunkshape=layers[i].shape,atom=tables.FloatAtom(),filters=tables.Filters(complevel=1))                
                hf_arr[:] = layers[i]
            hf.close()

        mask = np.round(layers[0]).astype(bool)

        img_extent = [lon.min(), lat.min(), lon.max(), lat.max()]

        lat_grid, lon_grid = np.meshgrid(lat*np.pi/180.,lon*np.pi/180.)
        x=np.dstack([lon_grid,lat_grid]+[l for l in layers[1:]])
        
        if x.shape[:-1] != mask.shape:
            raise RuntimeError, 'Shape mismatch in mask and other layers.'

        return mask, x, img_extent
    __all__ += ['make_covering_raster']
except:
    pass

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

    b.imshow(grid_convert(arr,'x+y+','y+x+'), interpolation='nearest')    
    # pl.colorbar()    
    plot_species(session, species[0], species[1], b, negs=True, **kwds)    
    return out, arr
    
def current_state_map(M, session, species, mask, x, img_extent, thin=1, **kwds):
    "Maps the current state of the model."

    out = pm.value(M.p)(x)

    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(grid_convert(arr,'x+y+','y+x+'), interpolation='nearest')    
    b.plot(pm.value(M.x_fr)[:,0]*180./np.pi, pm.value(M.x_fr)[:,1]*180./np.pi, 'g.', markersize=2)
    # b.drawcoastlines(color=(.9,.4,.5))
    # pl.colorbar()    

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