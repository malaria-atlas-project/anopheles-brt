from query import species_query, list_species
from models import Session

site_list, expert_opinion = species_query(Session(), 3)
for x in site_list:
    if sum(count or 0 for count in x[1:4]) != x[4]:
        print "Wrong."
