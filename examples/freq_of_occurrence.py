#!/usr/bin/python3

import os
import sys

# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')

import matplotlib

matplotlib.use('Agg')
import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np

import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print('available systems:', larda.connectors.keys())
print("available parameters: ", [(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())

data_loc = '/'

case_list = []
##

# new case study
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 13, 34, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 13, 40, 0),
#    'plot_range': [0, 10200],
#    'scatter_lim': [[-40, 0], [-4, 2]]})

case_list.append({
'begin_dt': datetime.datetime(2019, 2, 12, 7, 0, 0),
'end_dt': datetime.datetime(2019, 2, 12, 7, 45, 0),
'plot_range': [2000, 9000],
'scatter_lim':  [[-40, 0], [-4, 2]]})

for case in case_list:

    begin_dt = case['begin_dt']
    end_dt = case['end_dt']
    plot_range = case['plot_range']
    scatter_lim = case['scatter_lim']

    interval = {'time': [h.dt_to_ts(begin_dt), h.dt_to_ts(end_dt)], 'range': plot_range}

    time_height = f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + str(plot_range)

    if not os.path.isdir(data_loc + time_height): os.mkdir(data_loc + time_height)
    os.chdir(data_loc + time_height)

    """
    Create frequency of occurrence plot for reflectivity values
    """

    try:
        LIMRAD94_Ze = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], plot_range)
        LIMRAD94_SLv = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], plot_range)

        fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Ze, z_converter='lin2z')

        file_name = 'limrad_' + LIMRAD94_Ze['name'] + '_' + time_height + '.png'
        fig.savefig(file_name, dpi=250)
        print('save figure to :: ', file_name)

        sens_lim = np.mean(LIMRAD94_SLv['var'], axis=0)
        fig, ax = pyLARDA.Transformations.plot_frequency_of_ocurrence(LIMRAD94_Ze, x_lim=[-50, -10],
                                                                      sensitivity_limit=sens_lim,
                                                                      z_converter='lin2z')
        file_name = 'limrad_FOC_' + LIMRAD94_SLv['name'] + '_' + time_height + '.png'
        fig.savefig(file_name, dpi=250)
        print('save figure to :: ', file_name)

    except Exception as e:
        raise e

