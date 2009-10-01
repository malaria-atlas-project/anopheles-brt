import numpy as np
from numpy.testing import *
import nose,  warnings
from anopheles import extract_environment
import os
import anopheles
from map_utils import grid_convert, reconcile_multiple_rasters
import tables as tb
data_dirname = os.path.join(anopheles.__path__[0] ,'../datafiles')

def rotate(s,e,t):
    return s*np.cos(t) + e*np.sin(t), -s*np.sin(t) + e*np.cos(t)

def f(s, e, t=0, sc=1):
    out = np.empty((len(s), len(e)))
    for i in xrange(len(s)):
        for j in xrange(len(e)):
            x,y = rotate(s[i]/sc,e[j],t)
            out[i,j] = np.exp(-(x+y)**2*2)*np.cos(x*5) + np.exp(-np.abs(x))*np.sin(x*3)
    return out

def write_datafile(x,y,t=0,sc=1,view='x+y+'):
    hf = tb.openFile(os.path.join(data_dirname, 'test_extraction_%s.hdf5'%view),'w')
    hf.createArray('/','lon',x*180./np.pi)
    hf.createArray('/','lat',y*180./np.pi)
    d = grid_convert(f(x,y,t,sc), 'x+y+', view)
    hf.createCArray('/','data',shape=d.shape,chunkshape=(10,10),atom=tb.FloatAtom())
    hf.root.data[:] = d
    hf.root.data.attrs.view = view
    hf.close()
    return 'test_extraction_%s'%view, tb.openFile(os.path.join(data_dirname, 'test_extraction_%s.hdf5'%view))
    
x = np.linspace(-2,2,100)
y = np.linspace(-1,1,100)

n = 1000

import pylab as pl

class test_environment(object):
    
    def test_extraction(self):
        "Tests that extracting environmental layers works correctly for multiple views."
        for order in ['xy','yx']:
            for first in ['+','-']:
                for second in ['+','-']:
                    view = order[0]+first+order[1]+second
                    fname, hf = write_datafile(x,y,view=view)
                    
                    x_ind = np.random.randint(len(x)-2,size=n)+1
                    y_ind = np.random.randint(len(y)-2,size=n)+1

                    x_extract = x[x_ind]*180./np.pi + np.random.normal()*.001
                    y_extract = y[y_ind]*180./np.pi + np.random.normal()*.001
                    
                    e2 = grid_convert(hf.root.data[:], view, 'x+y+')[(x_ind,y_ind)]
                    
                    hf.close()

                    e1 = extract_environment(fname, np.vstack((x_extract,y_extract)).T, cache=False)
                    
                    assert_almost_equal(e1,e2,decimal=3)


    def test_reconciliation(self):
        "Tests that reconciling multiple rasters works correctly for multiple views."
        lims = {}
        ns = {}
        hfs = {}
        views = []
        for order in ['xy','yx']:
            for first in ['+','-']:
                for second in ['+','-']:
                    view = order[0]+first+order[1]+second
                    views.append(view)
                    
                    lim_lo = np.random.uniform(-1,-.5,size=2)
                    lim_hi = np.random.uniform(.5,1,size=2)
                    lim = [lim_lo[0], lim_hi[0], lim_lo[1], lim_hi[1]]
                    n = np.random.randint(200,500,size=2)
                    lims[view]=np.array(lim)*180./np.pi
                    ns[view]=n
                    
                    x = np.linspace(lim[0],lim[1],n[0])
                    y = np.linspace(lim[2],lim[3],n[1])
                    
                    print view
                    fname, hf = write_datafile(x,y,view=view)
                    hfs[view] = hf
                    
        lon, lat, layers = reconcile_multiple_rasters([hfs[v].root for v in views], thin=2)
        
        # Check that the layers all have the right shape
        assert(np.all([(l.shape == (len(lon),len(lat))) for l in layers]))
        
        # Check that the limits are inside the joint intersection
        assert(np.all([(lon.min() >= l[0]) * (lon.max() <= l[1]) * (lat.min() >= l[2]) * (lat.max() <= l[3]) for l in lims.itervalues()]))
        
        # Check that the coarseness is maximal
        assert([(lims[v][1]-lims[v][0])/float(ns[v][0]-1) <= lon[1]-lon[0] for v in views])
        assert([(lims[v][2]-lims[v][3])/float(ns[v][1]-1) <= lat[1]-lat[0] for v in views]) 
        
        lonmin = np.min([l[0] for l in lims.itervalues()])
        lonmax = np.max([l[1] for l in lims.itervalues()])
        latmin = np.min([l[2] for l in lims.itervalues()])
        latmax = np.max([l[3] for l in lims.itervalues()])
        
        import pylab as pl
        for i in xrange(len(views)):
            pl.figure(figsize=(10,6))
            pl.subplot(1,2,2)
            pl.imshow(grid_convert(layers[i],'x+y+','y+x+'),extent=[lon.min(),lon.max(),lat.min(),lat.max()])
            pl.title(views[i] + ': ' + str(hfs[views[i]].root.data.shape))
            pl.axis([lonmin, lonmax, latmin, latmax])
            pl.subplot(1,2,1)
            pl.imshow(grid_convert(hfs[views[i]].root.data[:], views[i], 'y+x+'), extent = lims[views[i]])
            pl.plot([lon.min(), lon.max(), lon.max(), lon.min()],[lat.min(), lat.min(), lat.max(), lat.max()],'r-')
            pl.axis([lonmin, lonmax, latmin, latmax])
        
        
    
                
if __name__ == '__main__':
    test_environment().test_reconciliation()
    # nose.runmodule()
