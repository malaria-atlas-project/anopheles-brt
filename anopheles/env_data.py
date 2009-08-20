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

import os
import anopheles
import map_utils
import hashlib
import tables as tb

__all__ = ['data_dir','extract_environment']

data_dirname = os.path.join(anopheles.__path__[0] ,'../datafiles')
data_dir = map_utils.LazyDataDirectory(data_dirname)

def extract_environment(name, x):
    
    x_hash = hashlib.sha1(x.data).hexdigest()
    fname = name + '_' + x_hash + '.hdf5'
    if 'anopheles-caches' in os.listdir('.'):
        if fname in os.listdir('anopheles-caches'):
            hf = tb.openFile(os.path.join('anopheles-caches',fname))
            return hf.root.eval[:]
    
    print 'Evaluation of environmental layer %s on array with SHA1 hash %s not found, recomputing.'%(name, hashlib.sha1(x.data).hexdigest())
            
    hr = data_dir.gethf(name)
    
    if hasattr(hr, 'lon'):
        grid_lon = hr.lon[:]
    else:
        grid_lon = hr.long[:]
    grid_lat = hr.lat[:]
    
    grid_data = hr.data[:]
    if hasattr(hr, 'mask'):
        grid_mask = hr.mask[:]
    else:
        grid_mask = None
    
    eval = map_utils.interp_geodata(grid_lon, grid_lat, grid_data, x[:,0], x[:,1], grid_mask)
    
    hf = tb.openFile(os.path.join('anopheles-caches',fname),'w')
    hf.createArray('/','eval',eval)
    hf.close()
    
    return eval