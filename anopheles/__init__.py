import os,sys
import anopheles
import anopheles_query
from anopheles_query import *
import species


for mod in ['query_to_rec','model','spatial_submodels','utils','env_data','mahalanobis_covariance','mapping','validation_metrics','constrained_mvn_sample','constraints','step_methods']:
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
