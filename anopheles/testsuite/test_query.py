from numpy.testing import *
import nose,  warnings
from anopheles import species_query, list_species, Session

class test_queries(object):
    site_list, expert_opinion = species_query(Session(), 3)
    for x in site_list:
        if sum(count or 0 for count in x[1:4]) != x[4]:
            print "Wrong."

if __name__ == '__main__':
    nose.runmodule()
