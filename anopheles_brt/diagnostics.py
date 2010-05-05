from __future__ import division

# Copyright (C) 2009  Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Throughout, p is predicted and a is actual

import numpy as np
from env_data import extract_environment

__all__ = ['compose','simple_assessments','roc','plot_roc','plot_roc_']

def compose(*fns):
    def composite_function(*a, **k):
        result = fns[-1](*a, **k)
        for f in fns[-2::-1]:
            result = f(*result)
        return result
    return composite_function

def proportion_correct(p, a):
    return np.sum(p==a)/float(len(a))
    
def false_positives(p, a):
    return np.sum(p*(True-a))/float(len(a))

def false_negatives(p, a):
    return np.sum((True-p)*a)/float(len(a))
    
def sensitivity(p, a):
    return np.sum(p*a)/float(np.sum(a))
    
def specificity(p, a):
    return np.sum((True-p)*(True-a))/float(np.sum(True-a))
    
def producer_accuracy(p, a):
    return sensitivity(p, a), specificity(p, a)
    
def consumer_accuracy(p, a):
    return np.sum(p*a)/float(np.sum(p)), np.sum((True-p)*(True-a))/float(np.sum(True-p))

def kappa(p, a):
    pa = np.sum(a)/len(a)
    pp = np.sum(p)/len(p)
    pagree = pa*pp + (1-pa)*(1-pp)
    return (np.sum(p==a)/float(len(a))-pagree)/float(1-pagree)
    
simple_assessments = [proportion_correct, false_positives, false_negatives, sensitivity, specificity, producer_accuracy, consumer_accuracy, kappa]
    
def roc(ps, a):
    """
    ps is a stack of classification vectors.
    """
    
    t = np.linspace(0,1,500)
    tp = [1]
    fp = [1]
    
    marginal_p = np.sum(ps,axis=0)/ps.shape[0]
    tot_neg = float(np.sum(True-a))
    tot_pos = float(np.sum(a))
    
    for i in xrange(len(t)):
        p = marginal_p>t[i]
        fp_here = np.sum((True-a)*p)/tot_neg
        if fp_here < 1:
            tp.append(np.sum(p*a)/tot_pos)
            fp.append(fp_here)
        if fp_here == 0:
            break
            
    fp = np.array(fp)
    tp = np.array(tp)
    
    AUC = -np.sum(np.diff(fp)*(tp[1:] + tp[:-1]))/2.    
    return fp, tp, AUC 
    
def plot_roc_(fp, tp, AUC):
    import pylab as pl
        
    pl.fill(np.concatenate((fp,[0,1])), np.concatenate((tp,[0,0])), facecolor=(.8,.8,.9), edgecolor='k', linewidth=1)
    pl.plot([0,1],[0,1],'k-.')
    
    pl.xlabel('False positive rate')
    pl.ylabel('True positive rate')
    
    if np.isnan(AUC):
        raise ValueError, 'AUC is NaN.'
    pl.title('AUC: %.3f'%AUC)
    pl.axis([0,1,0,1])

plot_roc = compose(plot_roc_, roc)    

# def plot_validation_(results):
#     import pylab as pl
#     pl.close('all')
#     for k in results.iterkeys():
#         pl.figure()
#         if k=='roc':
#             plot_roc_(*results['roc'])
#         elif k in ['producer_accuracy','consumer_accuracy']:
#             pl.hist(results[k][:,0])
#             pl.title('%s, false')
#             pl.figure()
#             pl.hist(results[k][:,1])
#             pl.title('%s, true'%k)
#         else:
#             pl.hist(results[k])
#             pl.title(k)


# if __name__ == '__main__':
#     import pymc as pm
#     import pylab as pl
#     n = 1000
#     a = pm.rbernoulli(.7,size=n).astype('bool')
#     p = pm.rbernoulli(.7,size=n).astype('bool')
#     
#     for s in simple_assessments:
#         print s.__name__, s(p, a)
#     
#     ps = pm.rbernoulli(.7,size=(100,n)).astype('bool')
#     for i in xrange(100):
#         if np.random.random()<.05:
#             ps[i,:]=a
#     pl.clf()
#     fp,tp,auc = roc(ps,a)
#     plot_roc(ps, a)