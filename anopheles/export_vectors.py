from django.db import transaction, models
from map.vector.models import NoData, Anopheline, SubSpecies, CollectionMethod, IdentificationMethod, ControlMethod, Presence, SamplePeriod, SiteSamplePeriod, Sample, Identification, AdminLog
from map.biblio.models import Source
import csv

SOURCE_FIELDS = ('enl_id', 'author_main', 'author_initials', 'year', 'report_type', 'published',)
SITE_FIELDS = ("site_id", "country", "full_name", "admin1_paper", "admin2_paper", "admin3_paper", "admin2_id","lat", "long", "lat2", "long2", "lat3", "long3", "lat4", "long4", "lat5", "long5", "lat6", "long6", "lat7", "long7", "lat8", "long8", "latlong_source", "bestguess_good", "bestguess_rough", "site_notes", "area_type", "rural_urban", "forest", "rice",)
PRESENCE_FIELDS = ("anopheline__abbreviation", "complex", "subspecies__abbreviation",)
SAMPLE_PERIOD_FIELDS = ("start_month", "end_month", "start_year", "end_year", "control_type__abbreviation", "sample_aggregate_check", "ASSI", "notes",)
SAMPLE_FIELDS = ("collection_method__abbreviation","count", )
IDENTIFICATION_FIELDS = ("identification_method",)
ADMIN_FIELDS = ("person",)

HEADER_FIELDS = ("ENL_ID","AUTHOR","Initials","YEAR","REPORT_TYPE","PUBLISHED","SITE_ID","COUNTRY","FULL_NAME","ADMIN1_PAPER","ADMIN2_PAPER","ADMIN3_PAPER","ADMIN2_ID","LAT","LONG","LAT2","LONG2","LAT3","LONG3","LAT4","LONG4","LAT5","LONG5","LAT6","LONG6","LAT7","LONG7","LAT8","LONG8","LATLONG_SOURCE","GOOD_GUESS","BAD_GUESS","SITE_NOTES","AREA_TYPE","RURAL-URBAN","FOREST","RICE","MONTH_STVEC","MONTH_ENVEC","YEAR_STVEC","YEAR_ENVEC","CONTROL_TYPE","ALLN","NOTES_ASSI","NOTES_VECTOR","SPECIES1","s.s./s.l.","SPECIES2","MOSSAMP_TECH1","N1","MOSSAMP_TECH2","N2","MOSSAMP_TECH3","N3","MOSSAMP_TECH4","N4","MOS_ID1 ","MOS_ID2","MOS_ID3","MOS_ID4","DEC_ID","DEC_CHECK","MAP_CHECK",)


class VectorRow(object):
    def __init__(self):
        self.ls = []

    def append_fields(self, obj, fieldnames):
        for field in fieldnames:
            self.append_attr(obj, field)
    def append_denormalized_fields(self, obj_list, fieldnames, count):
        for obj in obj_list:
            self.append_fields(obj, fieldnames)
        #need nfieldnames * 
        blank_fields = (count - len(obj_list)) * len(fieldnames)
        self.ls.extend([None for x in range(blank_fields)])

    def len(self):
        return len(self.ls)

    def append_blank_fields(self, n):
        self.ls.extend(['' for i in range(n)])

    def append_attr(self, obj, fieldname):
        fns = fieldname.split("__")
        if len(fns) == 1:
            self.ls.append(obj.__getattribute__(fns[0]))
        elif len(fns) == 2:
            newobj = obj.__getattribute__(fns[0])
            if newobj:
                self.ls.append(newobj.__getattribute__(fns[1]))
            else:
                self.ls.append(None)
        else:
            print "WTF: ", fns
            raise ValueError
            


sample_periods = SiteSamplePeriod.objects.all()[0:1000]
writer = csv.writer(open('/usr/public/temp/rectangular_database.csv', "w"))
writer.writerow(HEADER_FIELDS)

for sample_period in sample_periods:
    row = VectorRow()
    print sample_period
    row.append_fields(sample_period.site, SITE_FIELDS)
    row.append_fields(sample_period.vector_presence, PRESENCE_FIELDS)
    writer.writerow([unicode(s).encode("utf-8") for s in row.ls])
