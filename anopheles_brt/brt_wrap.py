import numpy as np
import os
from env_data import extract_environment
from query_to_rec import sites_as_ndarray
import hashlib
import warnings
import matplotlib
matplotlib.use('pdf')

from pylab import rec2csv

def df_to_ra(d):
    "Converts an R data fram to a NumPy record array."
    a = [np.array(c) for c in d.iter_column()]
    n = np.array(d.colnames)
    return np.rec.fromarrays(a, names=','.join(n))
    
def sites_and_env(session, species, layer_names, glob_name, glob_channels):
    """
    Queries the DB to get a list of locations. Writes it out along with matching 
    extractions of the requested layers to a temporary csv file, which serves the 
    dual purpose of caching the extraction and making it easier to get data into 
    the BRT package.
    """
    breaks, x, found, zero, others_found, multipoints = sites_as_ndarray(session, species)
    fname = hashlib.sha1(found.tostring()+\
            glob_name+'channel'.join([str(i) for i in glob_channels])+\
            'layer'.join(layer_names)).hexdigest()+'.csv'

    if fname in os.listdir('anopheles-caches'):
        pass
    else:
        # Makes list of (key, value) tuples
        env_layers = map(lambda ln: extract_environment(ln, x), layer_names)\
                + map(lambda ch: (os.path.basename(glob_name)+'_'+str(ch), extract_environment(glob_name,x,\
                    postproc=lambda d: d==ch)[1]), glob_channels)
                    
        # for i,l in enumerate(env_layers):
        #         if np.any(np.isnan(l[1])):
        #             xbad = x[np.where(np.isnan(l[1]))]
        #             np.savetxt('outland.csv',xbad)
        #             from mpl_toolkits import basemap
        #             b = basemap.Basemap(xbad[:,0].min(),xbad[:,1].min(),xbad[:,0].max(),xbad[:,1].max(), resolution='i')
        #             b.plot(xbad[:,0],xbad[:,1],'r.', markersize=2)
        #             b.drawcoastlines(linewidth=.5)
        #             b.drawcountries(linewidth=.5)
        #             import pylab as pl
        #             pl.savefig('outland.pdf')
        #             if raw_input('Layer %s evaluated to NaN in some places. Dumped outland.csv and outland.pdf. Want to see the points?') in ['y','Y','yes','Yes','YES']:
        #                 from map_utils import import_raster, grid_convert
        #                 lon,lat,data,rtype = import_raster('globcover5k_CLEAN','/Volumes/data/mastergrids/cleandata/Globcover')
        #                 dlon = lon[1]-lon[0]
        #                 dlat = lat[1]-lat[0]
        #                 b = basemap.Basemap(lon.min(),lat.min(),lon.max()+dlon,lat.max()+dlat)
        #                 pl.clf()
        #                 b.imshow(grid_convert(data,'y-x+','y+x+'),interpolation='nearest')
        #                 b.plot(xbad[:,0],xbad[:,1],'r.', markersize=2)
        #                 for pt in xbad:
        #                     pl.axis([pt[0]-.5,pt[0]+.5,pt[1]-.5,pt[1]+.5])
        #                     pl.title(pt)
        #                     pl.savefig('%s.pdf'%pt)
        #                     from IPython.Debugger import Pdb
        #                     Pdb(color_scheme='LightBG').set_trace() 
        #             raise RuntimeError, 'Sort yourself out.'
        
        arrays = [(found>0).astype('int')] + [l[1] for l in env_layers]
        names = ['found'] + [l[0] for l in env_layers]
        
        data = np.rec.fromarrays(arrays, names=','.join(names))
        nancheck = np.array([np.any(np.isnan(row.tolist())) for row in data])
        if np.any(nancheck):
            print 'There were some NaNs in the data, probably points in the sea'
        data = data[np.where(True-nancheck)]
        rec2csv(data, os.path.join('anopheles-caches',fname))
    return fname

# Notes on return values from gbm.fit:
# - gbm.call : List whose values are mix of vectors, scalars and booleans.
# - fitted : Probabilities under the observations, float(n).
# - fitted.vars : Variance of fitted, float(n).
# - residuals : Residuals of fitted values, float(n). (For Bernoulli, value-fitted)
# - contributions : Relative importance of residuals ?
# - self.statistics : List whose values are all scalar except calibration, which is a vector with named columns.
# - cv.statistics : List whose values are all scalar except a couple of vectors.
# - weights : Vector of ones, usually, int(n)
# - trees.fitted : Number of trees. Vector of ints.
# - training.loss.values,cv.values,cv.loss.ses: Vectors of floats
# - cv.loss.matrix, cv.roc.matrix: Matrices of floats.

# Code for pretty.gbm.tree, which displays the trees:
# 
# pretty.gbm.tree <- function(object,i.tree=1)
# {
#    if((i.tree<1) || (i.tree>length(object$trees)))
#    {
#       stop("i.tree is out of range. Must be less than ",length(object$trees))
#    }
#    else
#    {
#       temp <- data.frame(object$trees[[i.tree]])
#       names(temp) <- c("SplitVar","SplitCodePred","LeftNode",
#                        "RightNode","MissingNode","ErrorReduction",
#                        "Weight","Prediction")
#       row.names(temp) <- 0:(nrow(temp)-1)
#    }
#    return(temp)
# }
# If you unpack the trees, you'll get a (number of trees, 8, number of nodes) array.
# The 8 gives you c("SplitVar","SplitCodePred","LeftNode","RightNode","MissingNode",
#                   "ErrorReduction","Weight","Prediction")

# I _think_  the first row's SplitCode tells where the split is along the SplitVar axis,
# and the RightNode's and LeftNode's SplitCodes tell what the values are on either side
# side of the split. Not sure what MissingNode is about but we have no missing data.


def unpack_brt_results(brt_results, *names):
    namelist = np.array(brt_results.names)
    return map(lambda n: np.array(brt_results[np.where(namelist==n)[0][0]]), names)

def unpack_brt_trees(brt_results, layer_names, glob_name, glob_channels):
    
    all_names = map(os.path.basename,layer_names) + map(lambda ch: os.path.basename(glob_name) + '_' + str(ch), glob_channels)
    
    tree_matrix = unpack_brt_results(brt_results, 'trees')[0]
    nice_trees = []
    for split_var, split_code_pred, left_node, right_node, missing_node, error_reduction, weight, prediction in tree_matrix:
        new_tree = {'variable': all_names[int(split_var[0])],
                    'split_loc': split_code_pred[0]}
        for i in xrange(1,len(split_var)):
            if i==left_node[0]:
                new_tree['left_val'] = split_code_pred[int(i)]
            elif i==right_node[0]:
                new_tree['right_val'] = split_code_pred[int(i)]
        nice_trees.append(new_tree)

    nice_tree_dict = {}
    for v in all_names:
        v_array = map(lambda t: [t['split_loc'], t['left_val'], t['right_val']], filter(lambda t: t['variable']==v, nice_trees))
        nice_tree_dict[v] = np.rec.fromarrays(np.array(v_array).T, names='split_loc,left_val,right_val') if v_array else None
                
    return all_names, nice_tree_dict
    
class brt_evaluator(object):
    def __init__(self, all_names, nice_tree_dict):
        self.all_names = all_names
        self.nice_tree_dict = nice_tree_dict
    def __call__(self, **pred_vars):
        if set(pred_vars.keys()) != set(self.all_names):
            raise ValueError, "You haven't supplied all the predictors."
        for n in self.all_names:
            pass
            
    
def brt(session, species, layer_names, glob_name, glob_channels, gbm_opts):
    from rpy2.robjects import r
    import anopheles_brt
    r.source(os.path.join(anopheles_brt.__path__[0],'brt.functions.R'))
    
    fname = sites_and_env(session, species, layer_names, glob_name, glob_channels)
    
    base_argstr = 'data=read.csv("anopheles-caches/%s"), gbm.x=2:%i, gbm.y=1, family="bernoulli"'%(fname, len(layer_names)+len(glob_channels)+1)
    opt_argstr = ', '.join([base_argstr] + map(lambda t: '%s=%s'%t, gbm_opts.iteritems()))
    
    brt_results = r('gbm.step(%s)'%opt_argstr)
    
    return brt_results