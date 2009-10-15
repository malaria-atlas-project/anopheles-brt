import numpy as np
species_name = 'Anopheles arabiensis'

def elev_check(x,p):
    return np.sum(p[np.where(x>2000)])

def loc_check(x,p):
    outside_lat = np.abs(x[:,1]*180./np.pi)>50
    outside_lon = 0
    return np.sum(p[np.where(outside_lat + outside_lon)])

env = ['MODIS-hdf5/daytime-land-temp.mean.geographic.world.2001-to-2006',
        'MODIS-hdf5/daytime-land-temp.annual-amplitude.geographic.world.2001-to-2006',
        'MODIS-hdf5/daytime-land-temp.annual-phase.geographic.world.2001-to-2006',
        'MODIS-hdf5/daytime-land-temp.biannual-amplitude.geographic.world.2001-to-2006',
        'MODIS-hdf5/daytime-land-temp.biannual-phase.geographic.world.2001-to-2006',
        'MODIS-hdf5/evi.mean.geographic.world.2001-to-2006',
        'MODIS-hdf5/evi.annual-amplitude.geographic.world.2001-to-2006',
        'MODIS-hdf5/evi.annual-phase.geographic.world.2001-to-2006',
        'MODIS-hdf5/evi.biannual-amplitude.geographic.world.2001-to-2006',
        'MODIS-hdf5/evi.biannual-phase.geographic.world.2001-to-2006',
        'MODIS-hdf5/raw-data.elevation.geographic.world.version-5']

# cf = {'MODIS-hdf5/raw-data.elevation.geographic.world.version-5':elev_check}
cf = {'location'  :loc_check}
# cf = {}