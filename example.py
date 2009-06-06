from sqlalchemy.orm import join
from models import Anopheline, Site, Presence, SamplePeriod, Session

"""
Get all sites where gambiae is found.
Here we've filtered on name - could have just got all anophelines and looped over all 53, passing the anopheline object each time to query for sites.
"""
session = Session()
gambiae = session.query(Anopheline).filter(Anopheline.name.ilike("%gambiae%")).one()
sites = session.query(Site).join(Presence).filter(Presence.anopheline==gambiae)
#SQL issued to db here - session queries are lazy.
gambiae_site_list = sites.all()
