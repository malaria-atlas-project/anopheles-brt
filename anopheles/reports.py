from sqlalchemy.orm import join
from sqlalchemy.sql import func, exists, and_, not_
from models import Anopheline, Site, AdminUnit, Presence, SamplePeriod, Session
from sqlalchemygeom import *

session = Session()

"""
obtain a list of anophelines
"""
sampleperiod_subq = session.query(
    SamplePeriod.anopheline_id,
    func.count(func.distinct(SamplePeriod.site_id)).label('site_count'),
    func.count('*').label('sampleperiod_count')
    ).group_by(SamplePeriod.anopheline_id).subquery()

point_subq = session.query(
    SamplePeriod.anopheline_id,
    func.count('*').label('count')
    ).filter(Site.area_type=='point').filter(exists().where(SamplePeriod.site_id==Site.site_id)).filter(Anopheline.id==SamplePeriod.anopheline_id)

q = session.query(Anopheline.name,
    func.coalesce(sampleperiod_subq.c.site_count,0),
    #func.coalesce(select([func.count().label('point_count')]).where(sampleperiod_subq.c.anopheline_id=Anopheline.id).as_scalar(),0),
    func.coalesce(sampleperiod_subq.c.sampleperiod_count, 0)
    )

q = q.outerjoin((sampleperiod_subq, Anopheline.id==sampleperiod_subq.c.anopheline_id))
#q = q.outerjoin((point_subq, Anopheline.id==point_subq.c.anopheline_id))


sites_in_sea = session.query(Site).filter(and_(not_(func.intersects(Site.geom, AdminUnit.geom)),AdminUnit.admin_level_id=='0'))

world = session.query(Site)[0]



#positive_absences = session.query(Site

#select a.name, coalesce(ss.site_count, 0), coalesce(ss.temporally_unique, 0)
#from vector_anopheline a
#left join
#(
#select vp.anopheline_id, count(distinct(vp.site_id)) as site_count, count(vp.anopheline_id) as temporally_unique from vector_sampleperiod vsp inner join vector_presence vp
#on vsp.vector_presence_id = vp.id
#group by vp.anopheline_id
#)
#as ss
#on a.id = ss.anopheline_id
#) to /tmp/excel2.txt
