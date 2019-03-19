#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')

import matplotlib

matplotlib.use('Agg')
import pyLARDA
import pyLARDA.helpers as h
import datetime
import scipy.ndimage as spn

import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')


year = 2018
month = 12
day = 18
HH0 = 9
MM0 = 30

MM1 = 32
#this time interval in necessary to load back and forth, as they are in
#separate files
#it is not required to hit the exact time, as the full file is loaded internally

dt_begin = datetime.datetime(year, month, day, HH0, MM0, 0)
dt_end = datetime.datetime(year, month, day, HH0, MM1, 0)
plot_range = [0, 12000]

MIRA_rhi_Z = larda.read("MIRA", "rhi_Z", [dt_begin, dt_end], [0, 'max'])
h.pprint(MIRA_rhi_Z)
MIRA_rhi_elv = larda.read("MIRA", "rhi_elv", [dt_begin, dt_end])
h.pprint(MIRA_rhi_elv)

fig, ax = pyLARDA.Transformations.plot_timeheight(
            MIRA_rhi_Z, z_converter='lin2z')
fig.savefig('MIRA_rhi_scan_Z.png', dpi=250)

fig, ax = pyLARDA.Transformations.plot_timeseries(
            MIRA_rhi_elv)
fig.savefig('MIRA_rhi_scan_elv.png', dpi=250)
