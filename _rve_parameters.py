'''
N2 RVE Dynamics input file

Soubhik M., Boston University, 
Boston MA, 02215
'''

'''Mass of nuclei'''
m_nuclei1 = 14.0
m_nuclei2 = 14.0

'''Maximum simulation time (in fs)'''
t_max = 100

'''Time interval (in fs)'''
dt = 0.1 

'''Total number of grid points'''
NGrid = 2000

'''Kinetic energy of incoming electron in (ev)'''
nEkin = 100
Ekin_MAX = 5.0
Ekin_MIN = 1.0

'''Input Potential energy file pathname'''
PECFILE = './data/N2_PEC'

'''Output file path'''
pathout = './data/Cross-section/'