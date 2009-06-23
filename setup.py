from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('anopheles',parent_package=None,top_path=None)

config.add_extension(name='utils',sources=['anopheles/utils.f'])

config.packages = ["anopheles"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))