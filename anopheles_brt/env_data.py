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
import map_utils
import hashlib
import numpy

__all__ = ['extract_environment']

def extract_environment(layer_name, x, postproc=lambda x:x):
    "Expects ALL locations to be in decimal degrees."
    
    fname = hashlib.sha1(x.tostring()+layer_name).hexdigest()+'.npy'
    path, name = os.path.split(layer_name)
    name = os.path.splitext(name)[0]
    if fname in os.listdir('anopheles-caches'):
        return name, numpy.load(os.path.join('anopheles-caches',fname))
    else:    
        grid_lon, grid_lat, grid_data, grid_type = map_utils.import_raster(name,path)
        
        # Convert to centroids
        grid_lon += (grid_lon[1]-grid_lon[0])/2.
        grid_lat += (grid_lat[1]-grid_lat[0])/2.
        
        # Interpolate
        extracted = map_utils.interp_geodata(grid_lon, grid_lat, postproc(grid_data).data, x[:,0], x[:,1], grid_data.mask, chunk=None, view='y-x+', order=0)
        numpy.save(os.path.join('anopheles-caches',fname), extracted)
        return name, extracted
