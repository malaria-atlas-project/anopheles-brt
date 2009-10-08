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

import pymc
import numpy
import time
from pymc import InstantiationDecorators
np=numpy

__all__ = ['Constraint','constraint','LatchingMCMC']

class Constraint(pymc.Potential):
    
    def __init__(self, penalty_value=0, *args, **kwds):
        """logp should return a positive number or zero. If zero, the constraint is satisfied.
            If a positive number, it should be the degree of violation.
            
            Penalty value should be a negative number or zero. The log-probability will be the
            penalty value times the output of logp_fun."""
        pymc.Potential.__init__(self, *args, **kwds)
        self.open(penalty_value)
        self.penalty_value=penalty_value
        self.isopen = True
    
    def open(self, penalty_value):
        """If the constraint is not satisfied, self.logp will be penalty_value"""
        wrapper = lambda  penalty_value=penalty_value, *args, **kwds: penalty_value * self._logp_fun(*args,**kwds)
        self._logp = pymc.LazyFunction.LazyFunction(fun=wrapper,
                                    arguments = self.parents,
                                    ultimate_args = self.extended_parents,
                                    cache_depth = self._cache_depth)
        self._logp.force_compute()
        self.isopen = True
        self.penalty_value=penalty_value
        
    def close(self): 
        """If the constraint is not satisfied, self.logp will be -inf"""
        wrapper = lambda *args, **kwds: -numpy.inf if self._logp_fun(*args,**kwds) > 0 else 0
        self._logp = pymc.LazyFunction.LazyFunction(fun=wrapper,
                                    arguments = self.parents,
                                    ultimate_args = self.extended_parents,
                                    cache_depth = self._cache_depth)
        self._logp.force_compute()
        self.isopen = False
        self.penalty_value=-numpy.inf

        
def constraint(__func__ = None, **kwds):
    def instantiate_c(__func__):
        junk, parents = pymc.InstantiationDecorators._extract(__func__, kwds, keys, 'Potential', probe=False)
        return Constraint(parents=parents, **kwds)

    keys = ['logp']

    instantiate_c.kwds = kwds

    if __func__:
        return instantiate_c(__func__)

    return instantiate_c

class LatchingMCMC(pymc.MCMC):

    def __init__(self, *args, **kwds):
        pymc.MCMC.__init__(self, *args, **kwds)
        self.constraints = set(filter(lambda x:isinstance(x,Constraint), self.potentials))
    
    def print_constraints(self):
        for c in self.constraints:
            if c.isopen:
                print '%s: open, penalty value %f, violation %f'%(c.__name__, c.penalty_value, c.logp/c.penalty_value)
            else:
                try:
                    c.logp
                    print '%s: closed, satisfied.'%c.__name__
                except pymc.ZeroProbability:
                    print '%s: closed, violated.'%c.__name__
                    
    def _loop(self):
        # Set status flag
        self.status='running'

        # Record start time
        start = time.time()
        
        open_constraints = filter(lambda x:x.penalty_value != 0, self.constraints)
        all_closed = None
        i=-1

        try:
            while i < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    break
                    
                # Check constraints
                if all_closed is None:
                    for c in open_constraints:
                        if c.logp == 0:
                            if self.verbose > 0:
                                print 'Closing constraint %s.'%c.__name__
                            c.close()
                    open_constraints = filter(lambda x:x.isopen, open_constraints)
                    if len(open_constraints)==0:
                        all_closed = self._current_iter
                        i=0
                        print 'All constraints closed!'
                    if self.verbose > 1:
                        self.print_constraints()
            
                else:
                    i = self._current_iter - all_closed

                # Tune at interval
                if i and not (i % self._tune_interval) and self._tuning:
                    self.tune()

                if i == self._burn:
                    if self.verbose>0:
                        print 'Burn-in interval complete'
                    if not self._tune_throughout:
                        if self.verbose > 0:
                            print 'Stopping tuning due to burn-in being complete.'
                        self._tuning = False

                # Tell all the step methods to take a step
                for step_method in self.step_methods:
                    if self.verbose > 2:
                        print 'Step method %s stepping' % step_method._id
                    # Step the step method
                    step_method.step()
                
                if i % self._thin == 0 and i >= self._burn:
                    self.tally()

                if self._save_interval is not None:
                    if i % self._save_interval==0:
                        self.save_state()

                if not i % 10000 and i and self.verbose > 0:
                    per_step = (time.time() - start)/i
                    remaining = self._iter - i
                    time_left = remaining * per_step

                    print "Iteration %i of %i (%i:%02d:%02d remaining)" % (i, self._iter, time_left/3600, (time_left%3600)/60, (time_left%60))

                if not i % 1000:
                    self.commit()

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status='halt'

        if self.status == 'halt':
            self._halt()
        
if __name__ == '__main__':
    x = pymc.Normal('x',0,1)
    @constraint(penalty_value=-3)
    def c(x=x):
        return x if x>0 else 0
