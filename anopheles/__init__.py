import os,sys
import anopheles

for mod in ['query_to_rec','query','models','model','spatial_submodels','utils','env_data','mahalanobis_covariance','mapping']:
    try:
        exec('from %s import *'%mod)
    except ImportError:
        cls, inst, tb = sys.exc_info()
        print 'Failed to import %s. Error message:\n\t%s'%(mod,inst.message)
try:
    import utils
except ImportError:
    print 'Failed to import utils'
    
from testsuite import test

print 'Syncing datafiles with server.'
os.system('rsync -ax rsync://map1.zoo.ox.ac.uk/map_envlayers/anopheles/ %s'%os.path.join(anopheles.__path__[0],'../datafiles'))