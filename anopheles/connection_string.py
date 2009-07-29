from map import local_settings as ls

#connection_string = 'postgres://anand:8lj23lafagjf@localhost:5432/map_vector3'
#connection_string = 'postgres://will:\@&tolm00@localhost/map_vector3'
connection_string = 'postgres://will:%s@%s:%s/%s' % (ls.DATABASE_PASSWORD, ls.DATABASE_HOST, ls.DATABASE_PORT, ls.DATABASE_NAME)
