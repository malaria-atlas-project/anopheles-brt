import os,sys
import anopheles
import anopheles_query
import species


for mod in ['query_to_rec','model','spatial_submodels','utils','env_data','mahalanobis_covariance','mapping']:
    try:
        exec('from %s import *'%mod)
    except ImportError:
        cls, inst, tb = sys.exc_info()
        print 'Failed to import %s. Error message:\n\t%s'%(mod,inst.message)
try:
    import utils
except ImportError:
    print 'Failed to import utils'
    
#from testsuite import test
