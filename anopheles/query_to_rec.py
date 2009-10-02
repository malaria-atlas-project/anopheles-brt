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

import numpy as np


from anopheles_query import *
from map_utils import multipoly_sample
import tables as tb
import sys, os
import shapely

__all__ = ['site_to_rec', 'sitelist_to_recarray', 'list_species', 'species_query', 'map_extents', 'multipoint_to_ndarray', 'sample_eo', 'map_extents', 'point_to_ndarray']


def multipoint_to_ndarray(mp):
    "Converts a multipont to a coordinate array IN RADIANS."
    return np.array([[p.x, p.y] for p in mp.geoms])*np.pi/180.

def point_to_ndarray(p):
    return np.array([p.x, p.y])*np.pi/180.

def site_to_rec(s):
    """
    Converts a site to a flat x,y,n record.
    WARNING: Takes only the first point from multipoints.
    """
    n = 0 if s[1] is None else s[1]
    m = s[0]
    if m is None:
        return None
    if m.geoms._length > 0:
        raise ValueError, 'This is a multipoint.'
    else:
        p = m.geoms.next()
        return p.x, p.y, n
    
def sitelist_to_recarray(sl):
    """
    Converts a list of sites to a NumPy record array.
    WARNING: Takes only the first point from multipoints.
    """
    recs = filter(lambda x: x is not None, map(site_to_rec, sl))
    return np.rec.fromrecords(recs, names='x,y,n')
        
def map_extents(pos_recs, eo):
    "Figures out good extents for a basemap."
    return [min(pos_recs.x.min(), eo.bounds[0]),
            min(pos_recs.y.min(), eo.bounds[1]),
            max(pos_recs.x.max(), eo.bounds[2]),
            max(pos_recs.y.max(), eo.bounds[3])]
    
            
def sample_eo(session, species, n_in, n_out):
    sites, eo = species_query(session, species[0])
    
    fname = '%s_eo_pts_%i_%i.hdf5'%(species[1],n_in,n_out)

    if 'anopheles-caches' in os.listdir('.'):
        if fname in os.listdir('anopheles-caches'):
            hf = tb.openFile(os.path.join('anopheles-caches', fname))
            pts_in = hf.root.pts_in[:]
            pts_out = hf.root.pts_out[:]
        
            return pts_in, pts_out

    print 'Cached expert-opinion points not found, recomputing.'
    print 'Querying world'
    world = session.query(World)[0].geom
    print 'Differencing with expert opinion'
    not_eo = world.difference(eo)
    
    print 'Sampling outside'
    lon_out, lat_out = multipoly_sample(n_out, not_eo)
    print 'Sampling inside'
    lon_in, lat_in = multipoly_sample(n_in, eo)
    
    print 'Writing out'
    pts_in = np.vstack((lon_in, lat_in)).T*np.pi/180. 
    pts_out = np.vstack((lon_out, lat_out)).T*np.pi/180.
    
    hf = tb.openFile(os.path.join('anopheles-caches',fname),'w')
    hf.createArray('/','pts_in',pts_in)
    hf.createArray('/','pts_out',pts_out)
    
    return pts_in, pts_out

# Pylab-dependent stuff
try:
    import pylab as pl
    from mpl_toolkits import basemap
    from map_utils import plot_unit

    __all__ += ['split_recs', 'plot_species']

    def split_recs(recs):
        "Splits records into positive and negative versions."
        pos_recs = recs[np.where(recs.n>0)]
        neg_recs = recs[np.where(recs.n<=0)]
        return pos_recs, neg_recs
    
    def plot_species(session, species, name, b=None, negs=True, **kwds):
        "Plots the expert opinion, positives and not-observeds for the given species."
        sites, eo = species_query(session, species)
        ra = sitelist_to_recarray(sites)
        pos_recs, neg_recs = split_recs(ra)
        if b is None:
            b = basemap.Basemap(*map_extents(pos_recs, eo), **kwds)
        b.drawcoastlines(color='w',linewidth=.5)
        # b.drawcountries(color='w',linewidth=.5)
        pl.title(name, style='italic')        
        plot_unit(b, eo, '-', color=(.4,.4,.9), label='_nolegend_', linewidth=.75)        
        if negs:
            b.plot(neg_recs.x, neg_recs.y, '.', color=(.1,.6,.6), markersize=1.5, label='Observed')        
        b.plot(pos_recs.x, pos_recs.y, '.', color=(.9,.4,.4), markersize=1.5, label='Observed')


except ImportError:
    kls,inst,tb = sys.exc_info()
    print 'Warning, could not import Pylab. Original error message:\n\n' + inst.message

if __name__ == '__main__':
    session = Session()
    species = list_species(session)    
    pts_in, pts_out = sample_eo(session, species[1], 1000, 1000)
    sites, eo = species_query(session,species[1][0])
    # from map_utils import multipoly_sample
    # lon,lat = multipoly_sample(1000,eo)
    # 
    # ra = sitelist_to_recarray(sites)
    # pos_recs, neg_recs = split_recs(ra)
    # b = basemap.Basemap(*map_extents(pos_recs, eo))
    # plot_species(session, species[1][0], species[1][1], b=b) 
    # b.plot(lon,lat,'k.',markersize=1.5)
   
    # plot_species(session,44,'Anopheles punctimacula',resolution='l')
    # for s in species:
    #     pl.clf()
    #     try:
    #         plot_species(session,s[0],s[1],resolution='l')
    #     except IncompleteDataError:
    #         print '\n\n\n No EO for %s\n\n\n'%s[1]
