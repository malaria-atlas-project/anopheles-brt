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
    
    if sys.platform=='darwin':
        bin='/opt/local/bin'
    else:
        bin = '/usr/bin'
        
    file(os.path.join(bin,'anopheles-brt'),'w').write('#!%s/python\n\n'%bin+file('anopheles-brt').read())
    os.system('chmod ugo+x %s/anopheles-brt'%bin)