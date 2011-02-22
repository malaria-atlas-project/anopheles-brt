import os, sys
user = sys.argv[1]

os.system('ssh -L 2345:localhost:5432 -fN %s@map1.zoo.ox.ac.uk'%user)
