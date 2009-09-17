import numpy as np
from numpy.testing import *
import nose,  warnings
from anopheles import extract_environment
import os
import anopheles
from map_utils import grid_convert
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
    hf = tb.openFile(os.path.join(data_dirname, 'test_extraction.hdf5'),'w')
    hf.createArray('/','lon',x*180./np.pi)
    hf.createArray('/','lat',y*180./np.pi)
    d = grid_convert(f(x,y,t,sc), 'x+y+', view)
    hf.createCArray('/','data',shape=d.shape,chunkshape=(10,10),atom=tb.FloatAtom())
    hf.root.data[:] = d
    hf.root.data.attrs.view = view
    hf.close()
    return 'test_extraction', tb.openFile(os.path.join(data_dirname, 'test_extraction.hdf5'))
    
x = np.linspace(-2,2,100)
y = np.linspace(-1,1,100)

n = 1000

import pylab as pl

class test_extraction(object):
    
    def test1(self):
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

    
                
if __name__ == '__main__':
    # test_extraction().test1()
    nose.runmodule()
