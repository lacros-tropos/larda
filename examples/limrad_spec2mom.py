"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python LIMRAD94_to_Cloudnet.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""

import sys, time

sys.path.append('../')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpecToMom_Util as stm
import pyLARDA.NcWrite as nc

import datetime
import logging

import numpy as np


def make_container_from_spectra(spectra_all_chirps, variable, var_name):
    import copy

    container = dict()
    spectra = spectra_all_chirps[0]

    container['dimlabel'] = ['time', 'range']
    container['filename'] = spectra['filename']
    container['paraminfo'] = copy.deepcopy(spectra['paraminfo'])
    container['rg_unit'] = spectra['rg_unit']
    container['colormap'] = spectra['colormap']
    container['var_unit'] = ''
    container['var_lims'] = ''
    container['system'] = spectra['system']
    container['name'] = var_name
    container['rg'] = np.array([rg for ic in spectra_all_chirps for rg in ic['rg']])
    container['ts'] = spectra['ts']
    container['mask'] = np.isnan(variable[:])
    container['var'] = variable[:]

    return container


def calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt, nsd, png_path):
    AvgNum_in = larda.read("LIMRAD94", "AvgNum", [begin_dt, end_dt])
    DoppLen_in = larda.read("LIMRAD94", "DoppLen", [begin_dt, end_dt])
    MaxVel_in = larda.read("LIMRAD94", "MaxVel", [begin_dt, end_dt])

    # depending on how much files are loaded, AvgNum and DoppLen are multidimensional list
    if len(AvgNum_in['var'].shape) > 1:
        AvgNum = AvgNum_in['var'][0]
        DoppLen = DoppLen_in['var'][0]
        DoppRes = np.divide(2.0 * MaxVel_in['var'][0], DoppLen_in['var'][0])
    else:
        AvgNum = AvgNum_in['var']
        DoppLen = DoppLen_in['var']
        DoppRes = np.divide(2.0 * MaxVel_in['var'], DoppLen_in['var'])

    n_chirps = len(AvgNum)

    LIMRAD_Zspec = []
    for ic in range(n_chirps):
        LIMRAD_Zspec.append(larda.read("LIMRAD94", "C" + str(ic + 1) + "VSpec", [begin_dt, end_dt], [0, 'max']))
        LIMRAD_Zspec[ic].update({'no_av': np.divide(AvgNum[ic], DoppLen[ic]), 'DoppRes': DoppRes[ic]})

    # Logicals for different tasks
    include_noise = False

    # remove noise from raw spectra and calculate radar moments
    noise_est = stm.noise_estimation(LIMRAD_Zspec, n_std_deviations=nsd, include_noise=include_noise)

    #  dimensions:
    #       -   LIMRAD_Zspec[:]['var']      [Nchirps][ntime, nrange]
    #       -   LIMRAD_Zspec[:]['vel']      [Nchirps][nDoppBins]
    #       -   LIMRAD_Zspec[:]['DoppRes']  [Nchirps][1]
    #       -   noise_est[:]['signal']      [Nchirps][ntime, nrange]
    #       -   noise_est[:]['threshold']   [Nchirps][ntime, nrange]

    LIMRAD_moments = stm.spectra_to_moments_rpgfmcw94(LIMRAD_Zspec, noise_est)

    container = {'Ze': make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments['Ze_lin'].T, 'Ze'),
                 'VEL': make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments['VEL'].T, 'VEL'),
                 'sw': make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments['sw'].T, 'sw'),
                 'skew': make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments['skew'].T, 'skew'),
                 'kurt': make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments['kurt'].T, 'kurt')}

    return container


if __name__ == '__main__':

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo')
    c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

    print('available systems:', larda.connectors.keys())
    print("available parameters: ", [(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
    print('days with data', larda.days_with_data())

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    if 'date' in kwargs:
        date = '20190110'
        begin_dt = datetime.datetime.strptime(date + ' 12:30:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 14:15:08', '%Y%m%d %H:%M:%S')
    else:
        date = '20190110'
        begin_dt = datetime.datetime.strptime(date + ' 13:10:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 14:15:08', '%Y%m%d %H:%M:%S')

    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0
    png_path = 'spec2mom_diff_NF/'

    LIMRAD94_vars = calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt, std_above_mean_noise, png_path)

    variable_list = ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP',
                     'MaxVel', 'C1Range', 'C2Range', 'C3Range']

    for var in variable_list:
        print('variable :: ' + var)
        # if var in ["Ze", "VEL", 'sw', "skew", 'kurt', 'DiffAtt', 'ldr']:
        #    kwargs = {'interp_rg_join': True}
        # else:
        kwargs = {}
        LIMRAD94_vars.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'], **kwargs)})

    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_vars, png_path, time_frame='141005-141508')
