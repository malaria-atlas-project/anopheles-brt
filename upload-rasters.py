import os, sys
import imp
suff = imp.get_suffixes()[2]
dmod = imp.load_module(os.path.splitext(sys.argv[1])[0], file(sys.argv[1]), '.', suff)
user = sys.argv[2]

def listdir_noext(l):
    return map(lambda x: os.path.splitext(x)[0], os.listdir(l))

glob_path, glob_name = os.path.split(dmod.glob_name)
glob_path = os.path.join('rasters',glob_path)
os.mkdir(glob_path)


if glob_name not in listdir_noext(glob_path):
    os.system('scp %s@map1.zoo.ox.ac.uk:%s* %s'%(user,os.path.join('/srv/data/mastergrids/cleandata',dmod.glob_name),glob_path))

for l in dmod.layer_names:
    l_path, l_name = os.path.split(l)
    l_path = os.path.join('rasters',l_path)
    os.mkdir(l_path)
    if l_name not in listdir_noext(l_path):
        os.system('scp %s@map1.zoo.ox.ac.uk:%s* %s'%(user,os.path.join('/srv/data/mastergrids/cleandata',l),l_path))





