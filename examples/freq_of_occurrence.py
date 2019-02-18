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

import xlrd

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
excel_sheet_loc = '/tmp/pycharm_project_626/scripts_Willi/plots/scatter_case_studies/lowlvl_cumulus/'

##############
include_noise=True

excel_sheet = xlrd.open_workbook(excel_sheet_loc)
sheet = excel_sheet.sheet_by_index(0)
n_cases = sheet.nrows  # exclude header

case_list = []
for icase in range(1, n_cases):
    irow = sheet.row_values(icase)
    irow[:3] = [int(irow[i]) for i in range(3)]

    if irow[7] != 'ex':
        case_list.append({
            'begin_dt': datetime.datetime.strptime(str(irow[0]) + ' ' + str(irow[1]), '%Y%m%d %H%M%S'),
            'end_dt': datetime.datetime.strptime(str(irow[0]) + ' ' + str(irow[2]), '%Y%m%d %H%M%S'),
            'plot_range': [float(irow[3]), float(irow[4])],
            'MDF_name': irow[5],
            'noisefac': irow[6],
            'notes': irow[7]})


LIMRAD_params = {'var_list': ['VEL', 'sw', 'skew', 'kurt'],
                 'var_lims': [[-4, 2], [0, 1], [-0.2, 0.2], [0, 3]],
                 'colormap': 'jet',
                 'z_conv': ['none', 'none', 'none', 'none']}


for case in case_list:

    print('load case: ', case)

    try:
        begin_dt = case['begin_dt']
        end_dt = case['end_dt']
        plot_range = case['plot_range']

        interval = {'time': [h.dt_to_ts(begin_dt), h.dt_to_ts(end_dt)], 'range': plot_range}

        time_height_MDF = '{}_{}_'.format(begin_dt.strftime("%Y%m%d_%H%M%S"), end_dt.strftime("%H%M%S"))\
                        + '{}-{}_{}'.format(str(plot_range[0]), str(plot_range[1]), case['MDF_name'])
        if include_noise: time_height_MDF = time_height_MDF + '_withnoise'

        # create folder for pngs it doesn't exist already
        if not os.path.isdir(data_loc + time_height_MDF): os.mkdir(data_loc + time_height_MDF)
        os.chdir(data_loc + time_height_MDF)

        """
            Create frequency of occurrence plot for reflectivity values
        """
        # load first LIMRAD dict to gather some more information
        LIMRAD94_Ze = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], plot_range)

        # load range_offsets
        range_C1 = larda.read("LIMRAD94", "C1Range", [begin_dt, end_dt], plot_range)['var'].max()
        range_C2 = larda.read("LIMRAD94", "C2Range", [begin_dt, end_dt], plot_range)['var'].max()


        if not include_noise:
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

        # create time-height plot of reflectivity and save as png
        titlestring = 'LIMRAD94 Ze -- date: {}, MDF: {}, nf: {}'.format(case['begin_dt'].strftime("%Y-%m-%d"),
                                                                        case['MDF_name'], case['noisefac'])
        fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Ze,
                                                          range_interval=plot_range,
                                                          z_converter='lin2z',
                                                          title=titlestring)
        file_name = 'limrad_' + LIMRAD94_Ze['name'] + '_' + time_height_MDF + '.png'
        fig.savefig(file_name, dpi=250)
        print('save figure to :: ', file_name)

        # create frequency of occurrence plot of LIMRAD94 reflectivity and save as png
        fig, ax = pyLARDA.Transformations.plot_frequency_of_ocurrence(LIMRAD94_Ze, x_lim=[-70, 10],
                                                                      sensitivity_limit=sens_lim,
                                                                      range_offset=[range_C1, range_C2],
                                                                      z_converter='lin2z',
                                                                      title=titlestring)
        file_name = 'limrad_FOC_' + LIMRAD94_SLv['name'] + '_' + time_height_MDF + '.png'
        fig.savefig(file_name, dpi=250)
        print('save figure to :: ', file_name)

        LIMRAD94_Ze['var'] = h.lin2z(LIMRAD94_Ze['var'])  # convert to log units

        # load and save higher radar moments
        for var, i in zip(LIMRAD_params['var_list'], range(4)):
            LIMRAD94_var = larda.read("LIMRAD94", var, [begin_dt, end_dt], plot_range)
            LIMRAD94_var['var_lims'] = LIMRAD_params['var_lims'][i]
            if not include_noise: LIMRAD94_var['var'][combined_mask] = -999.0
            LIMRAD94_var['var'] = np.ma.masked_less_equal(LIMRAD94_var['var'], -999.0)

            titlestring = 'LIMRAD94 {} -- date: {}, MDF: {}, nf: {}'.format(var, case['begin_dt'].strftime("%Y-%m-%d"),
                                                                            case['MDF_name'], case['noisefac'])
            fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_var,
                                                              range_interval=plot_range,
                                                              z_converter=LIMRAD_params['z_conv'][i],
                                                              title=titlestring)
            file_name = 'limrad_' + var + '_' + time_height_MDF + '.png'
            fig.savefig(file_name, dpi=250)
            print('save figure to :: ', file_name)

            fig, ax = pyLARDA.Transformations.plot_scatter(LIMRAD94_Ze, LIMRAD94_var, identity_line=False,
                                                           title=titlestring)
            file_name = 'scatter_limrad_Ze_' + var + '_moments_' + time_height_MDF + '.png'
            print('save figure to :: ', file_name)
            fig.savefig(file_name, dpi=250)

    except Exception as e:
        print(e)
