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
from time import sleep
from pymc import InstantiationDecorators, utils
np=numpy
import sys

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
    
    def print_constraints(self, out=sys.stdout):
        for c in self.constraints:
            if c.isopen:
                print >> out, '%s: open, penalty value %f, violation %f'%(c.__name__, c.penalty_value, c.logp/c.penalty_value)
            else:
                try:
                    c.logp
                    print >> out, '%s: closed, satisfied.'%c.__name__
                except pymc.ZeroProbability:
                    print >> out, '%s: closed, violated.'%c.__name__

    def iprompt(self, out=sys.stdout):
        """Start a prompt listening to user input."""

        cmds = """
        Commands:
          i -- index: print current iteration index
          p -- pause: interrupt sampling and return to the main console.
                      Sampling can be resumed later with icontinue().
          h -- halt:  stop sampling and truncate trace. Sampling cannot be
                      resumed for this chain.
          c -- print constraints: print status of constraints we are trying
                      to satisfy.
          b -- bg:    return to the main console. The sampling will still
                      run in a background thread. There is a possibility of
                      malfunction if you interfere with the Sampler's
                      state or the database during sampling. Use this at your
                      own risk.
        """

        print >> out, """==============
    PyMC console
    ==============

        PyMC is now sampling. Use the following commands to query or pause the sampler.
        """
        print >> out, cmds

        prompt = True
        try:
            while self.status in ['running', 'paused']:
                    # sys.stdout.write('pymc> ')
                    if prompt:
                        out.write('pymc > ')
                        out.flush()

                    if self._exc_info is not None:
                        a,b,c = self._exc_info
                        raise a, b, c

                    cmd = utils.getInput().strip()
                    if cmd == 'i':
                        print >> out,  'Current iteration: ', self._current_iter
                        prompt = True
                    elif cmd == 'c':
                        self.print_constraints(out) 
                        prompt = True
                    elif cmd == 'p':
                        self.status = 'paused'
                        break
                    elif cmd == 'h':
                        self.status = 'halt'
                        break
                    elif cmd == 'b':
                        return
                    elif cmd == '\n':
                        prompt = True
                        pass
                    elif cmd == '':
                        prompt = False
                    else:
                        print >> out, 'Unknown command: ', cmd
                        print >> out, cmds
                        prompt = True

        except KeyboardInterrupt:
            if not self.status == 'ready':
                self.status = 'halt'


        if self.status == 'ready':
            print >> out, "Sampling terminated successfully."
        else:
            print >> out, 'Waiting for current iteration to finish...'
            while self._sampling_thread.isAlive():
                sleep(.1)
            print >> out, 'Exiting interactive prompt...'
            if self.status == 'paused':
                print >> out, 'Call icontinue method to continue, or call halt method to truncate traces and stop.'
                    
    def _loop(self):
        # Set status flag
        self.status='running'

        # Record start time
        start = time.time()
        
        open_constraints = filter(lambda x:x.isopen, self.constraints)
        if len(open_constraints)==0:
            all_closed = self._current_iter
            i=self._current_iter
        else:
            i=-1
            all_closed=None


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
                        print 'All constraints closed at iteration %i'%self._current_iter
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
