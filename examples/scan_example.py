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

dt_begin = datetime.datetime(year, month, day, HH0, MM0, 0)
dt_end = datetime.datetime(year, month, day, HH0, MM1, 0)
plot_range = [0, 12000]


#  RHI

#back and forth scanning are stored in separate files
#if the range from HH:30 to HH:32 is used, the files are joined
MIRA_rhi_Z = larda.read("MIRA", "rhi_Zg", [dt_begin, dt_end], [0, 'max'])
#or
MIRA_rhi_Z = larda.read("MIRA", "rhi_Zg", [dt_begin], [0, 'max'])
#it is not required to hit the exact time, as the full file is loaded internally
h.pprint(MIRA_rhi_Z)
# similarily for the elevation information
MIRA_rhi_elv = larda.read("MIRA", "rhi_elv", [dt_begin])
h.pprint(MIRA_rhi_elv)
MIRA_rhi_elv_2 = larda.read("MIRA", "rhi_elv", [dt_begin, dt_end])
MIRA_rhi_SLDR = larda.read("MIRA", "rhi_LDRg", [dt_begin, dt_end], [0, 'max'])


fig, ax = pyLARDA.Transformations.plot_timeheight(
            MIRA_rhi_Z, z_converter='lin2z')
fig.savefig('MIRA_rhi_scan_Z.png', dpi=250)

fig, ax = pyLARDA.Transformations.plot_timeseries(
            MIRA_rhi_elv)
fig.savefig('MIRA_rhi_scan_elv.png', dpi=250)

fig, ax = pyLARDA.Transformations.plot_rhi(MIRA_rhi_SLDR,
            MIRA_rhi_elv_2, z_converter='lin2z')
fig.savefig('MIRA_rhi_scan_SLDR.png', dpi=250)


# PPI
MM_PPI = 35 
dt = datetime.datetime(year, month, day, HH0, MM_PPI, 0)


MIRA_ppi_Z = larda.read("MIRA", "ppi_Zg", [dt_begin], [0, 'max'])
h.pprint(MIRA_ppi_Z)

MIRA_ppi_azi = larda.read("MIRA", "ppi_azi", [dt_begin])
h.pprint(MIRA_ppi_azi)

MIRA_ppi_vel = larda.read("MIRA", "ppi_VELg", [dt_begin], [0, 'max'])


fig, ax = pyLARDA.Transformations.plot_timeheight(
            MIRA_ppi_Z, z_converter='lin2z')
fig.savefig('MIRA_ppi_scan_Z.png', dpi=250)

fig, ax = pyLARDA.Transformations.plot_timeseries(
            MIRA_ppi_azi)
fig.savefig('MIRA_ppi_scan_azi.png', dpi=250)

fig, ax = pyLARDA.Transformations.plot_ppi(MIRA_ppi_vel, MIRA_ppi_azi, cmap='seismic')
fig.savefig('MIRA_ppi_scan_vel.png',dpi = 250)
