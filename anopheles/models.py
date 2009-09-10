# Copyright (C) 2009  William Temperley
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

from sqlalchemy import  Table, Column, Integer, String, Float, MetaData, ForeignKey, Boolean, create_engine
from sqlalchemygeom import Geometry
from sqlalchemy.orm import relation, backref, join, mapper, sessionmaker 
from connection_string import connection_string

engine = create_engine(connection_string, echo=False)
metadata = MetaData()
metadata.bind = engine
Session = sessionmaker(bind=engine)

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base(metadata=metadata)

__all__ = ['Anopheline', 'Anopheline2', 'Source', 'Site', 'ExpertOpinion', 'Presence', 'SamplePeriod', 'Session', 'World', 'Map', 'Collection', 'Identification','CollectionMethod', 'IdentificationMethod']

"""
update vector_anopheline set anopheline2_id = 
"""

class Anopheline(Base):
    """
    """
    __tablename__ = "vector_anopheline"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    abbreviation = Column(String)
    sub_genus = Column(String)
    species = Column(String)
    author = Column(String)
    is_complex  = Column(Boolean)
    def __repr__(self):
        return self.name

class Anopheline2(Base):
    """
    """
    __tablename__ = "vector_anopheline2"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    abbreviation = Column(String)
    sub_genus = Column(String)
    species = Column(String)
    author = Column(String)
    is_complex  = Column(Boolean)
    def __repr__(self):
        return self.name
    def get_scientific_name(self):
        species = self.species
        if self.is_complex:
            species += '*'
        name = "<i>Anopheles (%s) %s</i> %s" % (self.sub_genus, species, self.author)
        return name.replace("&", "&amp;")

class IdentificationMethod(Base):
    __table__ = Table('vector_identificationmethod', metadata, autoload=True)

class CollectionMethod(Base):
    __table__ = Table('vector_collectionmethod', metadata, autoload=True)

class Source(Base):
    __table__ = Table('source', metadata, autoload=True)

class Site(Base):
    """
    Represents a georeferenced site. The geometry returns a multipoint shapely object.
    The sample_periods property returns all sample periods linked to the site, aggregated across all studies.
    """
    __table__ = Table('site', metadata, Column('geom', Geometry(4326)), autoload=True)
    sample_periods = relation("SamplePeriod", backref="sites")

class SiteCoordinates(Base):
    """
    Represents a georeferenced site. The geometry returns a multipoint shapely object.
    The sample_periods property returns all sample periods linked to the site, aggregated across all studies.
    """
    __table__ = Table('site_coordinates', metadata, Column('geom', Geometry(4326)), autoload=True)
    site = relation("Site", backref="site_coordinates")

class ExpertOpinion(Base):
    __tablename__ = "vector_expertopinion"
    id = Column(Integer, primary_key=True)
    geom = Column(Geometry(4326))
    anopheline2_id = Column(Integer, ForeignKey('vector_anopheline2.id'))
    anopheline2 = relation(Anopheline2, backref="expert_opinion")

class Presence(Base):
    __tablename__ = "vector_presence"
    id = Column(Integer, primary_key=True)
    anopheline_id = Column(Integer, ForeignKey('vector_anopheline.id'))
    anopheline = relation(Anopheline, backref="presences")
    anopheline2_id = Column(Integer, ForeignKey('vector_anopheline2.id'))
    anopheline2 = relation(Anopheline2, backref="presences")
    subspecies_id = Column(Integer)
    site_id = Column(Integer, ForeignKey('site.site_id'))
    source_id = Column(Integer, ForeignKey('source.enl_id'))

class SamplePeriod(Base):
    """
    Represents a vector sample at a location. May have a specified time period.
    vector_site_sample_period is a view which aggregates samples across studies. 
    """
    #NB - A view 
    __tablename__ = "vector_sampleperiod"
    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey('site.site_id'))
    source_id = Column(Integer, ForeignKey('source.enl_id'))
    presence_id = Column(Integer)
    complex = Column(String)
    anopheline_id = Column(Integer, ForeignKey('vector_anopheline.id'))
    anopheline = relation(Anopheline, backref="sample_period")
    anopheline2_id = Column(Integer, ForeignKey('vector_anopheline2.id'))
    anopheline2 = relation(Anopheline2, backref="sample_period")
    start_month = Column(Integer, nullable=True)
    start_year = Column(Integer, nullable=True)
    end_month = Column(Integer, nullable=True)
    end_year = Column(Integer, nullable=True)

class Identification(Base):
    __table__ = Table('vector_identification', metadata, autoload=True)

class Collection(Base):
    """ 
    A set of mosquitos collected by a specific method 
    """ 
    __tablename__ = "vector_collection"
    id = Column(Integer, primary_key=True)
    ordinal = Column(Integer)
    count = Column(Integer)
    sample_period_id = Column(Integer, ForeignKey('vector_sampleperiod.id'))
    sample_period = relation(SamplePeriod, backref="sample")

class Region(Base):
    __table__ = Table('vector_region', metadata, autoload=True)

#class AdminUnit(Base):
#    """
#    Represents a vector sample at a location. May have a specified time period.
#    vector_site_sample_period is a view which aggregates samples across studies. 
#    """
#    __tablename__ = "adminunit"
#    id = Column(Integer, primary_key=True)
#    geom = Column(Geometry(4326))
#    admin_level_id = Column(String)
#    parent_id  = Column(Integer, ForeignKey('adminunit.id'))
#    name = Column(String)
#    dublin_core_id = Column(Integer)
#    gaul_code = Column(Integer)

class LayerStyle(Base):
    __tablename__ = "vector_layerstyle"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    fill_colour = Column(String)
    line_colour = Column(String)
    line_width = Column(String)
    opacity = Column(Float)
    def to_rgba(self, key):
        hex = self.__getattribute__(key)
        return float(int(hex[1:3], 16))/255,float(int(hex[3:5], 16))/255,float(int(hex[5:7], 16))/255,self.opacity

class Map(Base):
    __tablename__ = "vector_map"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    abbreviation = Column(String)
    sub_genus = Column(String)
    species = Column(String)
    author = Column(String)
    is_complex  = Column(Boolean)
    region_id = Column(Integer, ForeignKey('vector_region.id'))
    region = relation(Region, backref="vector_map")
    def get_scientific_name(self):
        species = self.species
        if self.is_complex:
            species += '*'
        name = "<i>Anopheles (%s) %s</i> %s" % (self.sub_genus, species, self.author)
        return name.replace("&", "&amp;")

class AnophelineLayer(Base):
    __tablename__ = "vector_anophelinelayer"
    id = Column(Integer, primary_key=True)
    ordinal = Column(Integer)
    map_id = Column(Integer, ForeignKey('vector_map.id'))
    map = relation(Map, backref=backref("anopheline_layers", order_by=ordinal))
    style_id = Column(Integer, ForeignKey('vector_layerstyle.id'))
    style = relation(LayerStyle, backref="anopheline_layer")
    layer_type = Column(String)
    is_presence = Column(Boolean, nullable=True)
    anopheline2_id = Column(Integer, ForeignKey('vector_anopheline2.id'))
    anopheline2 = relation(Anopheline2, backref="anopheline_layer")
    class Meta:
        ordering = ('ordinal',)

class World(Base):
    """
    The world as one big multipolygon
    """
    __tablename__ = "world"
    id = Column(Integer, primary_key=True)
    geom = Column(Geometry(4326))
