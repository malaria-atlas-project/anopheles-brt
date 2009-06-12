from sqlalchemy.orm import join
from sqlalchemy.sql import func, exists
from models import Anopheline, Site, Presence, SamplePeriod, Session
from sqlalchemygeom import *

"""
Get all sites where gambiae is found.
Here we've filtered on name - could have just got all anophelines and looped over all 53, passing the anopheline object each time to query for sites.
"""
session = Session()
mozzie = session.query(Anopheline).filter(Anopheline.name.ilike("%albimanus%")).one()

species_specific_subquery = session.query(SamplePeriod.site_id,func.count('*').label('sample_period_count')).filter(SamplePeriod.anopheline==mozzie).group_by(SamplePeriod.site_id).subquery()
any_anopheline = exists().where(SamplePeriod.site_id==Site.site_id)

#SQL issued to db here - session queries are lazy.
sites = session.query(Site.geom, species_specific_subquery.c.sample_period_count).outerjoin((species_specific_subquery, Site.site_id==species_specific_subquery.c.site_id)).filter(any_anopheline)
mozzie_site_list = sites.all()

from mpl_toolkits import basemap
from map_utils import *
import matplotlib
matplotlib.interactive(False)
b = basemap.Basemap(-128,-16,-30,35)
plot_unit(b,mozzie.expert_opinion[0].geom,'b-')
matplotlib.interactive(True)
b.drawcoastlines()
b.drawcountries()


#for eo in mozzie.expert_opinion:
#    print eo.geom.centroid 
#    print eo.geom.is_valid
