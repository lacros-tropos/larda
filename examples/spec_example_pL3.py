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

begin_dt = datetime.datetime(2018, 12, 14, 15, 40, 0)
end_dt = datetime.datetime(2018, 12, 14, 15, 50, 0)
plot_range = [0, 12000]


MIRA_Zspec=larda.read("MIRA","Zspec",[begin_dt,end_dt],[0,'max'])

#MIRA_noise = pyLARDA.helpers.noise_estimation(MIRA_Zspec, n_std_diviations=4.0)
# print(MIRA_Zspec)

# load LIMRAD spectra interpolated to velocity of lowest chirp
LIMRAD_Zspec = larda.read("LIMRAD94", "VSpec", [begin_dt, end_dt], [0, 'max'])

intervall = {'time': [h.dt_to_ts(begin_dt)],
             'range': [6000]}

print("slice range spectrogram")
range_spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD_Zspec, value=intervall)
range_spectrogram_MIRA   = pyLARDA.Transformations.slice_container(MIRA_Zspec,   value=intervall)

fig, ax = pyLARDA.Transformations.plot_spectra(range_spectrogram_LIMRAD, range_spectrogram_MIRA, z_converter='lin2z')

name = 'plots/PNG/spectra_limrad_mira_'
fig.savefig('plots/PNG/spectra_limrad_mira.png', dpi=250)


#print("slice time spectrogram")f
#time_interval = [h.dt_to_ts(datetime.datetime(2018, 12, 14, 9, 0)),
#                 h.dt_to_ts(datetime.datetime(2018, 12, 14, 9, 10))]
#
#time_spectrogram = pyLARDA.Transformations.slice_container(
#    LIMRAD_Zspec, value={'time': time_interval}, index={'range': [10, 11, 12]})
##
#print("slice single spectrum")
#single_spectrum = pyLARDA.Transformations.slice_container(
#    LIMRAD_Zspec, value={'time': [h.dt_to_ts(datetime.datetime(2018, 12, 14, 9, 0))]}, index={'range': [10]})
#print(single_spectrum)

# or load the single Chirps
#LIMRAD_Zspec = larda.read("LIMRAD94", "C1HSpec", [begin_dt, end_dt], [0, 'max'])
#LIMRAD_Zspec = larda.read("LIMRAD94", "C2HSpec", [begin_dt, end_dt], [0, 'max'])
#LIMRAD_Zspec = larda.read("LIMRAD94", "C3HSpec", [begin_dt, end_dt], [0, 'max'])
# print(LIMRAD_Zspec)
