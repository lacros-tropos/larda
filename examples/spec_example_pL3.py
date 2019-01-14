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

MIRA_Zspec = larda.read("MIRA", "Zspec", [begin_dt, end_dt], [0, 'max'])
LIMRAD_Zspec = larda.read("LIMRAD94", "VSpec", [begin_dt, end_dt], [0, 'max'])

LIMRAD_Zspec['var'] = LIMRAD_Zspec['var']/2.
MIRA_Zspec['var'] = MIRA_Zspec['var'][:,:,::-1]
#
# print("slice range spectrogram")
# intervall = {'time': [h.dt_to_ts(begin_dt)],
#             'range': [6000]}
# range_spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD_Zspec, value=intervall)
# range_spectrogram_MIRA   = pyLARDA.Transformations.slice_container(MIRA_Zspec,   value=intervall)
#
# fig, ax = pyLARDA.Transformations.plot_spectra(range_spectrogram_LIMRAD, range_spectrogram_MIRA, z_converter='lin2z')
# fig.savefig('plots/PNG/spectra_limrad_mira.png', dpi=250)


print("slice time-range spectrogram")
intervall = {'time': [h.dt_to_ts(begin_dt), h.dt_to_ts(datetime.datetime(2018, 12, 14, 16, 00, 0))],
             'range': [6000, 6100]}
spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD_Zspec, value=intervall)
spectrogram_MIRA = pyLARDA.Transformations.slice_container(MIRA_Zspec, value=intervall)

name = 'plots/PNG/spectra_limrad_mira_'
status = pyLARDA.Transformations.plot_multi_spectra(spectrogram_LIMRAD, spectrogram_MIRA, z_converter='lin2z', save=name)

