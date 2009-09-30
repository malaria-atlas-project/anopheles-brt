import pymc
import numpy
from pymc import InstantiationDecorators

__all__ = ['Constraint','constraint']

class Constraint(pymc.Potential):
    
    def __init__(self, penalty_value=0, *args, **kwds):
        pymc.Potential.__init__(self, *args, **kwds)
        self.open(penalty_value)
    
    def open(self, penalty_value):
        """If the constraint is not satisfied, self.logp will be penalty_value"""
        wrapper = lambda  penalty_value=penalty_value, *args, **kwds: penalty_value if self._logp_fun(*args,**kwds) else 0
        self._logp = pymc.LazyFunction.LazyFunction(fun=wrapper,
                                    arguments = self.parents,
                                    ultimate_args = self.extended_parents,
                                    cache_depth = self._cache_depth)
        self._logp.force_compute()
        
    def close(self): 
        """If the constraint is not satisfied, self.logp will be -inf"""
        self.open(-numpy.inf)
        
def constraint(__func__ = None, **kwds):
    def instantiate_c(__func__):
        junk, parents = pymc.InstantiationDecorators._extract(__func__, kwds, keys, 'Potential', probe=False)
        return Constraint(parents=parents, **kwds)

    keys = ['logp']

    instantiate_c.kwds = kwds

    if __func__:
        return instantiate_c(__func__)

    return instantiate_c
        
if __name__ == '__main__':
    x = pymc.Normal('x',0,1)
    @constraint(penalty_value=0)
    def c(x=x):
        return x>0
