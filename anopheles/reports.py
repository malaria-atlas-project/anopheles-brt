from sqlalchemy.orm import join
from sqlalchemy.sql import func, exists, and_, not_
from models import Anopheline, Site, AdminUnit, Presence, SamplePeriod, Session
from sqlalchemygeom import *

session = Session()

class RawQuery(object):
    """
    Emulates the behaviour of session.query
    used to enable lazy querying of raw sql
    """
    def __init__(self, session, sql):
        self.session = session
        self.sql = sql

    def all(self):
        result_proxy = self.session.execute(self.sql)
        return result_proxy.fetchall()

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
    func.coalesce(sampleperiod_subq.c.sampleperiod_count, 0)
    ).order_by(Anopheline.name.desc())

sites_and_sample_period_by_species = q.outerjoin((sampleperiod_subq, Anopheline.id==sampleperiod_subq.c.anopheline_id))


#sites_in_sea = session.query(Site).filter(and_(not_(func.intersects(Site.geom, AdminUnit.geom)),AdminUnit.admin_level_id=='0'))

sites_by_species_and_areatype = RawQuery(session,
"""
select
a.name,
(select count(*) from site where geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as all_sites,
(select count(*) from site where area_type = 'point' and geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as point_count,
(select count(*) from site where area_type = 'polygon small' and geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as polygon_small,
(select count(*) from site where area_type = 'polygon large' and geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as polygon_large,
(select count(*) from site where area_type = 'not specified' and geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as not_specified,
(select count(*) from site where area_type = 'wide area' and geom is not null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as wide_area,
(select count(*) from site where geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as all_sites_null_geom,
(select count(*) from site where area_type = 'point' and geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as point_count_null_geom,
(select count(*) from site where area_type = 'polygon small' and geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as polygon_small_null_geom,
(select count(*) from site where area_type = 'polygon large' and geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as polygon_large_null_geom,
(select count(*) from site where area_type = 'not specified' and geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as not_specified_null_geom,
(select count(*) from site where area_type = 'wide area' and geom is null and site_id in (select distinct(site_id) from vector_site_sample_period vsp where a.id = vsp.anopheline_id))as wide_area_null_geom
from vector_anopheline a
order by a.name desc;
"""
)

missing_coords = RawQuery(session,
"""
select s.site_id, min(source_id), count(*) as x 
from site s, vector_presence vp
where geom is null
and s.site_id = vp.site_id
group by s.site_id
order by x asc;
"""
)

bad_sequence = RawQuery(session,
"""
select distinct(source_id) 
from vector_presence where site_id in 
(select site_id as sum_ord from site_latlong
group by site_id
having not (
    (sum(ordinal) = 1 and count(*) = 1) or
    (sum(ordinal) = 3 and count(*) = 2) or
    (sum(ordinal) = 6 and count(*) = 3) or
    (sum(ordinal) = 10 and count(*) = 4) or
    (sum(ordinal) = 15 and count(*) = 5) or
    (sum(ordinal) = 21 and count(*) = 6) or
    (sum(ordinal) = 28 and count(*) = 7) or
    (sum(ordinal) = 36 and count(*) = 8))
)
;
"""
)
