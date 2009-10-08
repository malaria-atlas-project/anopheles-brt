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

import os, subprocess
import anopheles
import map_utils
import hashlib
import tables as tb
import warnings
import numpy as np

__all__ = ['get_datafile','extract_environment']

data_dirname = os.path.join(anopheles.__path__[0] ,'../datafiles')

def get_datafile(name):
    """Transparently synchronizes file with server (via ssh) and returns it."""
    data_root = os.path.join(anopheles.__path__[0],'../datafiles')
    path, name = os.path.split(name)
    data_path = os.path.join(data_root, path)
    fname = name+'.hdf5'
    data_file = os.path.join(data_path, fname)    
    remote_file = os.path.join('/srv/data',path,fname)
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    res = os.system('rsync --quiet map1.zoo.ox.ac.uk:%s %s'%(remote_file, data_file))
    if res > 0:
        warnings.warn('Failed to synchronize %s with server.'%name)
    
    return tb.openFile(data_file).root

def extract_environment(name, x, cache=True):
    
    """Expects ALL locations to be in decimal degrees."""
    
    if cache:
        x_hash = hashlib.sha1(x.data).hexdigest()
        fname = os.path.split(name)[1] + '_' + x_hash + '.hdf5'
        if 'anopheles-caches' in os.listdir('.'):
            if fname in os.listdir('anopheles-caches'):
                hf = tb.openFile(os.path.join('anopheles-caches',fname))
                out = hf.root.eval[:]
                hf.close()
                return out
    
        print 'Evaluation of environmental layer %s on array with SHA1 hash %s not found, recomputing.'%(name, hashlib.sha1(x.data).hexdigest())
    
    hr = get_datafile(name)
    
    if hasattr(hr, 'lon'):
        grid_lon = hr.lon[:]
    else:
        grid_lon = hr.long[:]
    grid_lat = hr.lat[:]
    
    grid_data = hr.data
    
    if hasattr(grid_data.attrs,'view'):
        view = grid_data.attrs.view
    else:
        raise ValueError, "Key 'view' not found in data array's attrs for datafile %s. \n\
    I could assume a default view, but you would suffer bitterly. \n\
    I could not bear it, human."%grid_data._v_file.filename
    
    if hasattr(hr, 'mask'):
        grid_mask = hr.mask
    else:
        grid_mask = None
    if np.prod(hr.data.shape) > 1e8:
        grid_chunk = hr.data.chunkshape
    else:
        grid_chunk=None
    
    eval = map_utils.interp_geodata(grid_lon, grid_lat, grid_data, x[:,0], x[:,1], grid_mask, grid_chunk, view=view)
    
    hr._v_file.close()
    
    if cache:
        hf = tb.openFile(os.path.join('anopheles-caches',fname),'w')
        hf.createArray('/','eval',eval)
        hf.close()
    
    return eval