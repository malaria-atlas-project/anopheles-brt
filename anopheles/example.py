from sqlalchemy.orm import join
from models import Anopheline, Site, Presence, SamplePeriod, Session

"""
Get all sites where gambiae is found.
Here we've filtered on name - could have just got all anophelines and looped over all 53, passing the anopheline object each time to query for sites.
"""
session = Session()
mozzie = session.query(Anopheline).filter(Anopheline.name.ilike("%albimanus%")).one()
sites = session.query(Site).join(Presence).filter(Presence.anopheline==mozzie)

#SQL issued to db here - session queries are lazy.
mozzie_site_list = sites.all()

#pointless example iterating over all geometries
for eo in mozzie.expert_opinion:
    print eo.geom.centroid 
    print eo.geom.is_valid
