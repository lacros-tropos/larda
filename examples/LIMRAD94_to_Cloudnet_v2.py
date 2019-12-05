
"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python spec2mom_limrad94.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""

import sys, datetime, time

sys.path.append('../')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.NcWrite as nc
from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container

import logging

import numpy as np

########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # Load LARDA

    # larda = pyLARDA.LARDA('remote', uri='http://larda.tropos.de/larda3').connect('lacros_dacapo', build_lists=False)
    larda = pyLARDA.LARDA().connect('lacros_dacapo_catalpa')
    c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

    # print('available systems:', larda.connectors.keys())
    # print("available parameters: ", [(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
    print('days with data', larda.days_with_data())

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    if 'date' in kwargs:
        date = str(kwargs['date'])
        begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')
    else:
        date = '20190918'
        begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')

    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    LIMRAD_Zspec = build_extended_container(larda, 'VSpec', begin_dt, end_dt,
                                            rm_precip_ghost=True,
                                            do_despeckle3d=False,
                                            estimate_noise=True,
                                            noise_factor=6.0
                                            )

    LIMRAD94_moments = spectra2moments(LIMRAD_Zspec, larda.connectors['LIMRAD94'].system_info['params'],
                                       despeckle=True,
                                       main_peak=True,
                                       filter_ghost_C1=True)

    ########################################################################################################################
    #
    #   _ _ _ ____ _ ___ ____    ____ ____ _    _ ___  ____ ____ ___ ____ ___     _  _ ____    ____ _ _    ____
    #   | | | |__/ |  |  |___    |    |__| |    | |__] |__/ |__|  |  |___ |  \    |\ | |       |___ | |    |___
    #   |_|_| |  \ |  |  |___    |___ |  | |___ | |__] |  \ |  |  |  |___ |__/    | \| |___    |    | |___ |___
    #
    for var in ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP', 'MaxVel', 'C1Range', 'C2Range', 'C3Range', 'SurfRelHum']:
        print('loading variable from LV1 :: ' + var)
        LIMRAD94_moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    LIMRAD94_moments['DiffAtt']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                            LIMRAD94_moments['DiffAtt']['var'])
    LIMRAD94_moments['ldr']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                        LIMRAD94_moments['ldr']['var'])

    cloudnet_remsens_lim_path = '/media/sdig/LACROS/cloudnet/data/'

    if 'path' in kwargs:
        path = kwargs['path']
    else:
        if c_info[0] == 'Punta Arenas':
            path = cloudnet_remsens_lim_path + 'punta-arenas-limrad/' + 'calibrated/limrad94/' + date[:4] + '/'
        elif c_info[0] == 'Leipzig':
            path = cloudnet_remsens_lim_path + 'leipzig/' + 'calibrated/limrad94/' + date[:4] + '/'
        elif c_info[0] == 'RV-Meteor':
            path = cloudnet_remsens_lim_path + 'leipzig/' + 'calibrated/limrad94/' + date[:4] + '/'
        else:
            print('Error: No other sites implemented jet!')
            sys.exit(-42)

    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_moments, path)

    ########################################################################################################################

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))

