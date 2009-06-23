from numpy.testing import *
import nose,  warnings
from anopheles import species_query, list_species, Session

class test_queries(object):
    def test_checksums(self):
        "Tests that the entries of the records add up."
        site_list, expert_opinion = species_query(Session(), 3)
        for x in site_list:
            assert(sum(count or 0 for count in x[1:4]) == x[4])
                
    def some_other_test(self):
        pass

if __name__ == '__main__':
    nose.runmodule()
