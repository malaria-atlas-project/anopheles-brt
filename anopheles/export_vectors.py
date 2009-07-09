from django.db import transaction, models
from map.vector.models import NoData, Anopheline, SubSpecies, CollectionMethod, IdentificationMethod, ControlMethod, Presence, SamplePeriod, SiteSamplePeriod, Sample, Identification, AdminLog
from map.biblio.models import Source
import csv

SOURCE_FIELDS = ('enl_id', 'author_main', 'author_initials', 'year', 'report_type', 'published',)
SITE_FIELDS = ("site_id", "country", "full_name", "admin1_paper", "admin2_paper", "admin3_paper", "admin2_id","lat", "long", "lat2", "long2", "lat3", "long3", "lat4", "long4", "lat5", "long5", "lat6", "long6", "lat7", "long7", "lat8", "long8", "latlong_source", "bestguess_good", "bestguess_rough", "vector_site_notes", "area_type", "rural_urban", "forest", "rice",)
PRESENCE_FIELDS = ("anopheline__abbreviation", "complex", "subspecies__abbreviation",)
SAMPLE_PERIOD_FIELDS = ("start_month", "start_year", "end_month", "end_year", "control_type__abbreviation", "sample_aggregate_check", "ASSI", "notes",)
SAMPLE_FIELDS = ("collection_method__abbreviation","count", )
IDENTIFICATION_FIELDS = ("identification_method",)
ADMIN_FIELDS = ("person",)

HEADER_FIELDS = ("enl_id",
"author",
"initials",
"year",
"report_type",
"published",
"site_id",
"country",
"full_name",
"admin1_paper",
"admin2_paper",
"admin3_paper",
"admin2_id",
"lat", "long", "lat2", "long2", "lat3", "long3", "lat4", "long4", "lat5", "long5", "lat6", "long6", "lat7", "long7", "lat8", "long8",
"latlong_source",
"good_guess",
"bad_guess",
"site_notes",
"area_type",
"rural-urban",
"forest",
"rice",
"species1",
"s.s./s.l.",
"species2",
"start_month",
"start_year",
"end_month",
"end_year",
"control_type",
"alln",
"notes_assi",
"notes_vector",
"sampling_technique1",
"n1",
"sampling_technique2",
"n2",
"sampling_technique3",
"n3",
"sampling_technique4",
"n4",
"mos_id1 ",
"mos_id2",
"mos_id3",
"mos_id4",
"dec_id",
"dec_check",
"map_check",)


class VectorRow(object):
    def __init__(self):
        self.ls = []

    def append_fields(self, obj, fieldnames):
        for field in fieldnames:
            self.append_attr(obj, field)
    def append_denormalized_fields(self, obj_list, fieldnames, count):
        for obj in obj_list:
            self.append_fields(obj, fieldnames)
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

site_sample_periods = SiteSamplePeriod.objects.all()[0:1000]
writer = csv.writer(open('/tmp/rectangular_database.csv', "w"))
writer.writerow(HEADER_FIELDS)

for site_sample_period in site_sample_periods:
    row = VectorRow()
    row.append_fields(site_sample_period.vector_presence.source, SOURCE_FIELDS)
    row.append_fields(site_sample_period.site, SITE_FIELDS)
    row.append_fields(site_sample_period.vector_presence, PRESENCE_FIELDS)
    row.append_fields(site_sample_period, SAMPLE_PERIOD_FIELDS)

    sample_period = SamplePeriod.objects.get(id=site_sample_period.id)
    row.append_denormalized_fields(sample_period.sample_set.all(), SAMPLE_FIELDS, 4)
    row.append_denormalized_fields(sample_period.identification_set.all(), IDENTIFICATION_FIELDS, 4)
    row.append_denormalized_fields(sample_period.adminlog_set.all(), ADMIN_FIELDS, 3)
    writer.writerow([unicode(s).encode("utf-8") for s in row.ls])
