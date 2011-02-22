from setuptools import setup
import sys
import os
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('anopheles_brt',parent_package=None,top_path=None)

config.add_extension(name='treetran', sources=['anopheles_brt/treetran.f'])

config.packages = ["anopheles_brt"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))    

    bin = os.path.expanduser('~/bin')
    executable = os.path.join(bin,'anopheles-brt')    
    python_executable = os.popen('which python').read()
    file(executable,'w').write('#!%s\n\n'%python_executable+file('anopheles-brt').read())
    os.system('chmod ugo+x %s'%executable)
