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
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print('available systems:', larda.connectors.keys())
print("available parameters: ", [(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())

data_loc = '/'

case_list = []
##############

# 20190207 12:34 - 13:05 UTC
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 12, 35, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 12, 38, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 12, 39, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 12, 43, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 12, 45, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 12, 48, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 12, 49, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 12, 54, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 12,55, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 12, 59, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 13, 0, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 13, 4, 0)})

## 20190207 14:09 - 15:10 UTC
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 9, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 14, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 15, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 19, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 20, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 24, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 25, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 29, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 35, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 39, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 40, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 44, 0)})
#case_list.append({
#    'begin_dt': datetime.datetime(2019, 2, 7, 14, 45, 0),
#    'end_dt': datetime.datetime(2019, 2, 7, 14, 50, 0)})

case_list.append({
    'begin_dt': datetime.datetime(2019, 2, 7, 14, 54, 0),
    'end_dt': datetime.datetime(2019, 2, 7, 14, 55, 50)})



LIMRAD_params = {'var_list': ['VEL', 'sw', 'skew', 'kurt'],
                 'var_lims': [[-4, 2], [0, 1], [-0.2, 0.2], [0, 3]],
                 'colormap': 'jet',
                 'z_conv': ['none', 'none', 'none', 'none']}


for case in case_list:

    begin_dt = case['begin_dt']
    end_dt = case['end_dt']

    # load first LIMRAD dict to gather some more information
    LIMRAD94_Ze = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], [0, 'max'])
    plot_range = [0, round(LIMRAD94_Ze['rg'][-1], -2)]  # round to next 100th meters

    interval = {'time': [h.dt_to_ts(begin_dt), h.dt_to_ts(end_dt)], 'range': plot_range}

    time_height = f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + str(plot_range)

    if not os.path.isdir(data_loc + time_height): os.mkdir(data_loc + time_height)
    os.chdir(data_loc + time_height)

    """
        Create frequency of occurrence plot for reflectivity values
    """

    # load range_offsets
    range_C1 = larda.read("LIMRAD94", "C1Range", [begin_dt, end_dt], plot_range)['var'].max()
    range_C2 = larda.read("LIMRAD94", "C2Range", [begin_dt, end_dt], plot_range)['var'].max()

    # load mira for filtering out noise pixels
    MIRA_Ze = larda.read("MIRA", "Zg", [begin_dt, end_dt], plot_range)
    MIRA_Ze_interp = pyLARDA.Transformations.interpolate2d(MIRA_Ze,
                                                           new_time=LIMRAD94_Ze['ts'],
                                                           new_range=LIMRAD94_Ze['rg'])

    combined_mask = np.logical_or(LIMRAD94_Ze['mask'], MIRA_Ze_interp['mask'])

    LIMRAD94_Ze['var'][combined_mask] = -999.0

    # load sensitivity limits (time, height) and calculate the mean over time
    LIMRAD94_SLv = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], plot_range)
    sens_lim = np.mean(LIMRAD94_SLv['var'], axis=0)

    # create time-height plot of reflectivity and save to png
    fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Ze, range_interval=plot_range, z_converter='lin2z')
    file_name = 'limrad_' + LIMRAD94_Ze['name'] + '_' + time_height + '.png'
    fig.savefig(file_name, dpi=250)
    print('save figure to :: ', file_name)

    # create frequency of occurrence plot of LIRMAD94 reflectivity and save to png
    fig, ax = pyLARDA.Transformations.plot_frequency_of_ocurrence(LIMRAD94_Ze, x_lim=[-70, 10],
                                                                  sensitivity_limit=sens_lim,
                                                                  range_offset=[range_C1, range_C2],
                                                                  z_converter='lin2z')
    file_name = 'limrad_FOC_' + LIMRAD94_SLv['name'] + '_' + time_height + '.png'
    fig.savefig(file_name, dpi=250)
    print('save figure to :: ', file_name)

    LIMRAD94_Ze['var'] = h.lin2z(LIMRAD94_Ze['var'])  # convert to log units

    # load and save higher radar moments
    for var, i in zip(LIMRAD_params['var_list'], range(4)):
        LIMRAD94_var = larda.read("LIMRAD94", var, [begin_dt, end_dt], plot_range)
        LIMRAD94_var['var_lims'] = LIMRAD_params['var_lims'][i]
        LIMRAD94_var['var'][combined_mask] = -999.0
        LIMRAD94_var['var'] = np.ma.masked_less_equal(LIMRAD94_var['var'], -999.0)

        fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_var,
                                                          range_interval=plot_range,
                                                          z_converter=LIMRAD_params['z_conv'][i])
        file_name = 'limrad_' + var + '_' + time_height + '.png'
        fig.savefig(file_name, dpi=250)
        print('save figure to :: ', file_name)

        fig, ax = pyLARDA.Transformations.plot_scatter(LIMRAD94_Ze, LIMRAD94_var, identity_line=False)
        file_name = 'scatter_limrad_Ze_'+var+'_higher_moments_' + time_height + '.png'
        print('save figure to :: ', file_name)
        fig.savefig(file_name, dpi=250)

