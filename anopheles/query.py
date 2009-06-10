# You need to supply connection.py ... my version has a password in it.
from connection import conn
from map_utils import NonSuckyShapefile
import os

cur = conn.cursor()
# cur.execute("select * from vector_presence")
# results = cur.fetchall()
# 
# print results

def data_and_eo(species, dir='../afro shapefiles020609'):
    fnames = os.listdir(dir)
    fnames = filter(lambda x: os.path.splitext(x)[1]=='.dbf', fnames)
    fnames = filter(lambda x: x.find(species)>-1, fnames)
    if len(fnames) > 1:
        raise ValueError, 'Multiple matches: \n\t- %s'%('\n\t- '.join(fnames))
    
    return NonSuckyShapefile(os.path.join(dir, os.path.splitext(fnames[0])[0]))    
    
if __name__ == '__main__':
    data_and_eo('arabiensis')
    # data_and_eo('anopheles')    