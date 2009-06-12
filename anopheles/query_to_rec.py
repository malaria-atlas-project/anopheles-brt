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
from sqlalchemy.orm import join
from sqlalchemy.sql import func, exists, and_, not_
from models import Anopheline, Site, Presence, SamplePeriod, Session
from sqlalchemygeom import *
import sys

__all__ = ['IncompleteDataError', 'site_to_rec', 'sitelist_to_recarray', 'list_species', 'species_query', 'map_extents']

class IncompleteDataError(BaseException):
    pass

def site_to_rec(s):
    "Converts a site to a flat x,y,n record"
    n = 0 if s[1] is None else s[1]
    # m is a MultiPoint
    m = s[0]
    if m is None:
        return None
    if m.geoms._length > 0:
        return None
    else:
        p = m.geoms.next()
        return p.x, p.y, n
    
def sitelist_to_recarray(sl):
    "Converts a list of sites to a NumPy record array"
    recs = filter(lambda x: x is not None, map(site_to_rec, sl))
    return np.rec.fromrecords(recs, names='x,y,n')
    
def list_species(session):
    return [(o.id, o.name) for o in Session().query(Anopheline)]
    
def species_query(session, species):
    """
    Takes a species string and returns two things: a NumPy record array
    of x,y,n records and a Shapely MultiPolygon object containing the 
    expert opinion.
    """
    mozzie = session.query(Anopheline).filter(Anopheline.id == species).one()

    species_specific_subquery = session.query(SamplePeriod.site_id,func.count('*').label('sample_period_count')).filter(and_(SamplePeriod.anopheline==mozzie, not_(func.coalesce(SamplePeriod.sample_aggregate_check, 1)==0))).group_by(SamplePeriod.site_id).subquery()
    any_anopheline = exists().where(SamplePeriod.site_id==Site.site_id)

    #SQL issued to db here - session queries are lazy.
    sites = session.query(Site.geom, species_specific_subquery.c.sample_period_count).outerjoin((species_specific_subquery, Site.site_id==species_specific_subquery.c.site_id)).filter(any_anopheline)
    mozzie_site_list = sites.all()
    
    if len(mozzie.expert_opinion)==0:
        raise IncompleteDataError

    return sitelist_to_recarray(mozzie_site_list), mozzie.expert_opinion[0].geom
    
def map_extents(pos_recs, eo):
    "Figures out good extents for a basemap."
    return [min(pos_recs.x.min(), eo.bounds[0]),
            min(pos_recs.y.min(), eo.bounds[1]),
            max(pos_recs.x.max(), eo.bounds[2]),
            max(pos_recs.y.max(), eo.bounds[3])]
            

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
    
    def plot_species(session, species, name, negs=True, **kwds):
        "Plots the expert opinion, positives and not-observeds for the given species."
        ra, eo = species_query(session, species)
        pos_recs, neg_recs = split_recs(ra)
        b = basemap.Basemap(*map_extents(pos_recs, eo), **kwds)
        b.drawcoastlines(color='w',linewidth=.5)
        b.drawcountries(color='w',linewidth=.5)
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
    # plot_species(session,44,'Anopheles punctimacula',resolution='l')
    for s in species:
        pl.clf()
        try:
            plot_species(session,s[0],s[1],resolution='l')
        except IncompleteDataError:
            print '\n\n\n No EO for %s\n\n\n'%s[1]
