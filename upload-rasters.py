import os, sys
import imp
suff = imp.get_suffixes()[2]
dmod = imp.load_module(os.path.splitext(sys.argv[1])[0], file(sys.argv[1]), '.', suff)
user = sys.argv[2]

def listdir_noext(l):
    return map(lambda x: os.path.splitext(x)[0], os.listdir(l))
    
def multimkdir(dir):
    if os.path.exists(dir):
        return
    head, tail = os.path.split(dir)
    if not os.path.exists(head):
        multimkdir(head)
    os.mkdir(tail)

glob_path, glob_name = os.path.split(dmod.glob_name)
glob_path = os.path.join('rasters',glob_path)
multimkdir(glob_path)


if glob_name not in listdir_noext(glob_path):
    s = os.system('scp %s@map1.zoo.ox.ac.uk:%s* %s'%(user,os.path.join('/srv/data/mastergrids/cleandata',dmod.glob_name),glob_path))
    if s != 0:
        raise KeyboardInterrupt()

for l in dmod.layer_names:
    l_path, l_name = os.path.split(l)
    l_path = os.path.join('rasters',l_path)
    multimkdir(l_path)
    if l_name not in listdir_noext(l_path):
        s = os.system('scp %s@map1.zoo.ox.ac.uk:%s* %s'%(user,os.path.join('/srv/data/mastergrids/cleandata',l),l_path))
        if s != 0:
            raise KeyboardInterrupt()





