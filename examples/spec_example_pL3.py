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
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())
# print("array_avail()", larda.array_avail(2015, 6))
# print("single month with new interface ", larda.instr_status(2015, 6))

# begin_dt=datetime.datetime(2018,12,2,16,0,0)
# end_dt=datetime.datetime(2018,12,2,22,0,0)
# plot_range = [300, 10000]

year = 2018
month = 12
day = 18
HH0 = 9
MM0 = 24

HH1 = 9
MM1 = 25

begin_dt = datetime.datetime(year, month, day, HH0, MM0, 0)
end_dt = datetime.datetime(year, month, day, HH1, MM1, 0)
plot_range = [0, 12000]

MIRA_Zspec = larda.read("MIRA", "Zspec", [begin_dt, end_dt], [0, 'max'])
LIMRAD_Zspec = larda.read("LIMRAD94", "VSpec", [begin_dt, end_dt], [0, 'max'])

print("slice time-range spectrogram")
interval = {'time': [h.dt_to_ts(begin_dt), h.dt_to_ts(end_dt)], 'range': [6300, 6400]}

spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD_Zspec, value=interval)
spectrogram_MIRA = pyLARDA.Transformations.slice_container(MIRA_Zspec, value=interval)

name = 'plots/PNG/spectra_limrad_mira_'
fig, ax = pyLARDA.Transformations.plot_spectra(spectrogram_LIMRAD, spectrogram_MIRA, z_converter='lin2z', save=name)

