"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python LIMRAD94_to_Cloudnet.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""

import sys, datetime

sys.path.append('../')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpecToMom_Util as stm
import pyLARDA.NcWrite as nc

import logging

import numpy as np


def make_container_from_spectra(spectra_all_chirps, values, paraminfo):
    """
    This routine will generate a larda container from calculated moments from spectra.

    Args:
        spectra_all_chirps (dict): dimension [nchirps], containing the spectrum values of the 94 GHz RPG cloud radar
        values (numpy array): dimension [nrange, ntimes], values of calculated moments
        paraminfo (dict): information from params_[campaign].toml for the specific variable

    Return:
        container (dict): larda data container
    """
    import copy

    spectra = spectra_all_chirps[0]

    container = {'dimlabel': ['time', 'range'], 'filename': spectra['filename'], 'paraminfo': copy.deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'], 'colormap': paraminfo['colormap'], 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'], 'system': paraminfo['system'], 'name': paraminfo['paramkey'],
                 'rg': np.array([rg for ic in spectra_all_chirps for rg in ic['rg']]), 'ts': spectra['ts'],
                 'mask': np.isnan(values[:]), 'var': values[:]}

    return container


def calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt, nsd, paraminfo):
    """
    This routine calculates the radar moments: reflectivity, mean Doppler velocity, spectrum width, skewness and
    kurtosis from the level 0 spectrum files of the 94 GHz RPG cloud radar.

    Args:
        begin_dt (datetime object): beginning of the time frame
        end_dt (datetime object): end of the time frame
        nsd (float): number of standard deviations above mean noise threshold
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Return:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

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

    cum_rg = [0]
    LIMRAD_Zspec = []
    for ic in range(n_chirps):
        LIMRAD_Zspec.append(larda.read("LIMRAD94", "C{}VSpec".format(ic + 1), [begin_dt, end_dt], [0, 'max']))
        LIMRAD_Zspec[ic].update({'no_av': np.divide(AvgNum[ic], DoppLen[ic]), 'DoppRes': DoppRes[ic]})
        cum_rg.append(cum_rg[ic] + LIMRAD_Zspec[ic]['rg'].size)

    # Logicals for different tasks
    include_noise = True if nsd < 0.0 else False

    # remove noise from raw spectra and calculate radar moments
    noise_est = stm.noise_estimation(LIMRAD_Zspec, n_std_deviations=nsd, include_noise=include_noise)

    #  dimensions:
    #       -   LIMRAD_Zspec[:]['var']      [Nchirps][ntime, nrange]
    #       -   LIMRAD_Zspec[:]['vel']      [Nchirps][nDoppBins]
    #       -   LIMRAD_Zspec[:]['DoppRes']  [Nchirps][1]
    #       -   noise_est[:]['signal']      [Nchirps][ntime, nrange]
    #       -   noise_est[:]['threshold']   [Nchirps][ntime, nrange]

    LIMRAD_moments = stm.spectra_to_moments_rpgfmcw94(LIMRAD_Zspec, noise_est, include_noise=include_noise)

    stm.noise_pixel_filter(LIMRAD_moments, cum_rg)

    container_dict = {}
    for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        container_dict.update({mom: make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments[mom].T, paraminfo[mom])})

    return container_dict


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
        date = '20190324'
        begin_dt = datetime.datetime.strptime(date + ' 2:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 2:15:08', '%Y%m%d %H:%M:%S')

    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0
    png_path = 'spec2mom_diff_NF/'

    LIMRAD94_vars = calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt, std_above_mean_noise,
                                                             larda.connectors['LIMRAD94'].system_info['params'])

########################################################################################################################
#
#   Plotting radar moments for quicklooks
#
#    #LIMRAD94_ZE = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], [0, 'max'])
#    LIMRAD94_ZE = LIMRAD94_vars['Ze']
#    LIMRAD94_ZE['var_unit'] = 'dBZ'
#    fig, axZe = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE, fig_size=[16, 3], range_interval=[0, 12000],
#                                                        z_converter='lin2z', title='nofilter')
#    fig_name = 'limrad_Ze' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '.png'
#    fig.savefig(fig_name, dpi=250)
#
#    LIMRAD94_VEL = LIMRAD94_vars['VEL']
#    fig, axZe = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, fig_size=[16, 3], range_interval=[0, 12000])
#    fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '.png'
#    fig.savefig(fig_name, dpi=250)
#
#    LIMRAD94_sw = LIMRAD94_vars['sw']
#    fig, axZe = pyLARDA.Transformations.plot_timeheight(LIMRAD94_sw, fig_size=[16, 3], range_interval=[0, 12000])
#    fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '.png'
#    fig.savefig(fig_name, dpi=250)
########################################################################################################################



########################################################################################################################
#
#   Plotting sensitivity limits for different dates
#
#    # load sensitivity limits (time, height) and calculate the mean over time
#    begin_dt = datetime.datetime.strptime(date + ' 1:00:05', '%Y%m%d %H:%M:%S')
#    end_dt = datetime.datetime.strptime(date + ' 13:00:08', '%Y%m%d %H:%M:%S')
#    LIMRAD94_SLv_nf5 = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], [0, 12000])
#    sens_lim_nf5 = np.mean(LIMRAD94_SLv_nf5['var'], axis=0)
#
#    date = '20190102'
#    begin_dt = datetime.datetime.strptime(date + ' 0:00:05', '%Y%m%d %H:%M:%S')
#    end_dt = datetime.datetime.strptime(date + ' 23:50:08', '%Y%m%d %H:%M:%S')
#    LIMRAD94_SLv_nf6 = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], [0, 12000])
#    sens_lim_nf6 = np.mean(LIMRAD94_SLv_nf6['var'], axis=0)
#
#    import matplotlib.pyplot as plt
#
#    fig, ax = plt.subplots(1, figsize=(13, 8))
#
#    sens_lim_nf5 = h.lin2z(sens_lim_nf5)
#    sens_lim_nf6 = h.lin2z(sens_lim_nf6)
#
#    y_bins_nf5 = LIMRAD94_SLv_nf5['rg']
#    y_bins_nf6 = LIMRAD94_SLv_nf6['rg']
#    ax.plot(sens_lim_nf5, y_bins_nf5, linewidth=2.0, color='red', label='sensitivity limit noisefactor 5')
#    ax.plot(sens_lim_nf6, y_bins_nf6, linewidth=2.0, color='blue', label='sensitivity limit noisefactor 6')
#
#    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
#    plt.grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
#    plt.legend(loc='upper left')
#
#    ax.set_xlim([-60, -20])
#    ax.set_ylim([0, 12000])
#
#    fig_name = 'limrad_senslimit_nf5_nf6' + '0-12000' + '.png'
#    fig.savefig(fig_name, dpi=250)
########################################################################################################################


########################################################################################################################
#
#   Generating calibrated netcdf file
#
#    for var in ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP', 'MaxVel', 'C1Range', 'C2Range', 'C3Range']:
#        print('variable :: ' + var)
#        # if var in ["Ze", "VEL", 'sw', "skew", 'kurt', 'DiffAtt', 'ldr']:
#        #    kwargs = {'interp_rg_join': True}
#        # else:
#        kwargs = {}
#        LIMRAD94_vars.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'], **kwargs)})
#
#    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_vars, png_path, time_frame='141005-141508')
########################################################################################################################
