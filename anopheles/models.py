from sqlalchemy import  Table, Column, Integer, String, MetaData, ForeignKey, create_engine
from sqlalchemygeom import Geometry
from sqlalchemy.orm import relation, join
from sqlalchemy.orm import sessionmaker
from connection_string import connection_string

engine = create_engine(connection_string, echo=True)
metadata = MetaData()
metadata.bind = engine
Session = sessionmaker(bind=engine)

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Anopheline(Base):
    """
    """
    __tablename__ = "vector_anopheline"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    abbreviation = Column(String)
    def __repr__(self):
        return self.name

class Site(Base):
    """
    Represents a georeferenced site. The geometry returns a multipoint shapely object.
    The sample_periods property returns all sample periods linked to the site, aggregated across all studies.
    """
    __tablename__ = "site"
    site_id = Column(Integer, primary_key=True)
    sample_periods = relation("SamplePeriod", backref="sites")
    geom = Column(Geometry(4326))
    vector_full_name = Column(String)

class ExpertOpinion(Base):
    __tablename__ = "vector_expertopinion"
    id = Column(Integer, primary_key=True)
    geom = Column(Geometry(4326))
    anopheline_id = Column(Integer, ForeignKey('vector_anopheline.id'))
    anopheline = relation(Anopheline, backref="expert_opinion")

class Presence(Base):
    __tablename__ = "vector_presence"
    id = Column(Integer, primary_key=True)
    anopheline_id = Column(Integer, ForeignKey('vector_anopheline.id'))
    anopheline = relation(Anopheline, backref="presences")
    subspecies_id = Column(Integer)
    site_id = Column(Integer, ForeignKey('site.site_id'))

class SamplePeriod(Base):
    """
    Represents a vector sample at a location. May have a specified time period.
    vector_site_sample_period is a view which aggregates samples across studies. 
    """
    __tablename__ = "vector_site_sample_period"
    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey('site.site_id'))
    anopheline_id = Column(Integer, ForeignKey('vector_anopheline.id'))
    anopheline = relation(Anopheline, backref="sample_period")
    start_month = Column(Integer, nullable=True)
    start_year = Column(Integer, nullable=True)
    end_month = Column(Integer, nullable=True)
    end_year = Column(Integer, nullable=True)
    sample_aggregate_check = Column(Integer, nullable=True)


class AdminUnit(Base):
    """
    Represents a vector sample at a location. May have a specified time period.
    vector_site_sample_period is a view which aggregates samples across studies. 
    """
    __tablename__ = "gis_adminunit"
    id = Column(Integer, primary_key=True)
    geom = Column(Geometry(4326))
    admin_level_id = Column(String)
    parent_id  = Column(Integer, ForeignKey('gis_adminunit.id'))
    name = Column(String)
    dublin_core_id = Column(Integer)
    gaul_code = Column(Integer)
