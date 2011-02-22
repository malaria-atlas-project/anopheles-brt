import os, sys

mod = os.path.splitext(sys.argv[1])[0]
user = sys.argv[2]

exec('import %s'%mod)
exec('dmod = %s'%mod)
glob_name = os.path.split(dmod.glob_name)[1]
rasters_here = map(lambda x: os.path.splitext(x)[0], os.listdir('rasters'))

if glob_name not in rasters_here:
    os.system('scp %s@map1.zoo.ox.ac.uk:%s* rasters'%(user,os.path.join('/srv/data/mastergrids/cleandata',dmod.glob_name)))

for l in dmod.layer_names:
    if os.path.split(l)[1] not in rasters_here:
        os.system('scp %s@map1.zoo.ox.ac.uk:%s* rasters'%(user,os.path.join('/srv/data/mastergrids/cleandata',l)))
