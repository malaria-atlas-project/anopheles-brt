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

__all__ = ['presence_map','current_state_map','current_state_slice','subset_x']

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

def subset_x(mask, x, img_extent, new_extent):
    lonmin = img_extent[0]
    latmin = img_extent[1]
    dlon = (img_extent[2]-lonmin)/float(x.shape[0])
    dlat = (img_extent[3]-latmin)/float(x.shape[1])
    
    lonind = (int((new_extent[0]-lonmin)/dlon), int((new_extent[2]-lonmin)/dlon))
    latind = (int((new_extent[1]-latmin)/dlat), int((new_extent[3]-latmin)/dlat))
    
    subset_mask = mask[lonind[0]:lonind[1],latind[0]:latind[1]]
    subset_x = x[lonind[0]:lonind[1],latind[0]:latind[1],:]
    subset_extent = np.array([subset_x[:,:,0].min(), subset_x[:,:,1].min(), subset_x[:,:,0].max(), subset_x[:,:,1].max()])*180./np.pi
    
    return subset_mask, subset_x, subset_extent

def presence_map(M, session, species, burn=0, thin=1, trace_thin=1, **kwds):
    "Converts the trace to a map of presence probability."
    
    from mpl_toolkits import basemap
    import pylab as pl
    import time
    
    chain_len = len(M.db._h5file.root.chain0.PyMCsamples)
    
    mask, x, img_extent = make_covering_raster(thin, M.env_variables, **kwds)

    out = np.zeros(mask.shape)

    time_count = -np.inf
    time_start = time.time()
    
    ptrace = M.trace('p')[:]
    for i in xrange(burn, len(M.trace('p_find')[:]), trace_thin):
        
        if time.time() - time_count > 10:
            print (((i-burn)*100)/(chain_len)), '% complete',
            if i>burn:
                time_count = time.time()      
                print 'expect results '+time.ctime((time_count-time_start)*(chain_len-burn)/float(i-burn)+time_start)
            else:
                print
        
        p = ptrace[i]
        pe = p(x)
        out += pe/float(chain_len-burn)
    
    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(grid_convert(arr,'x+y+','y+x+'), interpolation='nearest')    
    # pl.colorbar()    
    plot_species(session, species[0], species[1], b, negs=True, **kwds)    
    return out, arr
    
def current_state_map(M, session, species, mask, x, img_extent, thin=1, f2p=None, **kwds):
    "Maps the current state of the model."

    out = pm.utils.value(M.p)(x, f2p=f2p)

    b = basemap.Basemap(*img_extent)
    arr = np.ma.masked_array(out, mask=True-mask)

    b.imshow(grid_convert(arr,'x+y+','y+x+'), interpolation='nearest')    
    # b.plot(pm.value(M.x_fr)[:,0]*180./np.pi, pm.value(M.x_fr)[:,1]*180./np.pi, 'g.', markersize=2)
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