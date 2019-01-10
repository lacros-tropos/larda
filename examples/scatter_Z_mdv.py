#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')

import matplotlib

matplotlib.use('Agg')

import pyLARDA
import datetime

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())

# begin_dt=datetime.datetime(2018,12,6,0,1)
begin_dt = datetime.datetime(2018, 12, 6, 1, 40, 0)
end_dt   = datetime.datetime(2018, 12, 6, 4, 0,  0)

plot_range = [300, 10000]

# load the reflectivity data
MIRA_Zg = larda.read("MIRA", "Zg", [begin_dt, end_dt], [0, 'max'])
MIRA_Zg['var_lims'] = [-60, 20]

LIMRAD94_Z = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], [0, 'max'])
LIMRAD94_Z['var_lims'] = [-60, 20]

LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])

fig, ax = pyLARDA.Transformations.scatter(MIRA_Zg, LIMRAD94_Z_interp, var_lim=[-75, 20],
                                          custom_offset_lines=5.0, z_converter='lin2z')

fig.savefig('scatter_mira_limrad.png', dpi=250)
