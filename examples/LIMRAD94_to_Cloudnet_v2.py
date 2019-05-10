
"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python limrad_spec2mom.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""

import sys, datetime, time, toml

sys.path.append('../')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpecToMom_Util as s2m
import pyLARDA.NcWrite as nc

import logging

import numpy as np


def make_container_from_spectra(spectra_all_chirps, values, paraminfo, invalid_mask):
    """
    This routine will generate a larda container from calculated moments from spectra.

    Args:
        spectra_all_chirps (list of dicts): dimension [nchirps], containing the spectrum
                                            values of the 94 GHz RPG cloud radar
        values (numpy array): dimension [nrange, ntimes], values of calculated moments
        paraminfo (dict): information from params_[campaign].toml for the specific variable

    Return:
        container (dict): larda data container
    """
    import copy

    spectra = spectra_all_chirps[0]

    container = {'dimlabel': ['time', 'range'], 'filename': spectra['filename'], 'paraminfo': copy.deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'], 'colormap': paraminfo['colormap'],
                 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'],
                 'system': paraminfo['system'], 'name': paraminfo['paramkey'],
                 'rg': np.array([rg for ic in spectra_all_chirps for rg in ic['rg']]), 'ts': spectra['ts'],
                 'mask': invalid_mask, 'var': values[:]}

    return container


def calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt, nsd, paraminfo, **kwargs):
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

    #  dimensions:
    #       -   LIMRAD_Zspec[:]['var']      [Nchirps][ntime, nrange]
    #       -   LIMRAD_Zspec[:]['vel']      [Nchirps][nDoppBins]
    #       -   LIMRAD_Zspec[:]['no_av']    [Nchirps]
    #       -   LIMRAD_Zspec[:]['DoppRes']  [Nchirps]
    rg_offsets = [0]
    LIMRAD_Zspec = []
    for ic in range(n_chirps):
        tstart = time.time()
        LIMRAD_Zspec.append(larda.read("LIMRAD94", "C{}VSpec".format(ic + 1), [begin_dt, end_dt], [0, 'max']))
        LIMRAD_Zspec[ic].update({'no_av': np.divide(AvgNum[ic], DoppLen[ic]), 'DoppRes': DoppRes[ic]})
        rg_offsets.append(rg_offsets[ic] + LIMRAD_Zspec[ic]['rg'].size)
        print('reading spectra, chirp = {}, elapsed time = {:.3f} sec.'.format(ic + 1, time.time() - tstart))

    ######################################################################
    #
    # 3rd chirp ghost echo filter
    #if 'filter_ghost_C3' in kwargs: s2m.filter_ghost_echos_RPG94GHz_FMCW(LIMRAD_Zspec, kwargs['filter_ghost_C3'])

    if 'filter_ghost_C3' in kwargs and kwargs['filter_ghost_C3']:
        tstart = time.time()
    #
        # threholds for 3rd chrip ghost echo filter
        c3_vel_max = 2.5
        c3_Ze_max = h.z2lin(-22.5)
        # c3_Ze_max = h.z2lin(-15.5)
    #
        idx_left = np.argwhere(-c3_vel_max > LIMRAD_Zspec[2]['vel']).max()
        idx_right = np.argwhere(c3_vel_max < LIMRAD_Zspec[2]['vel']).min()
    #
        Ze_lin_left = LIMRAD_Zspec[2]['var'][:, :, :idx_left].copy()
        Ze_lin_right = LIMRAD_Zspec[2]['var'][:, :, idx_right:].copy()
    #
        # if noise was already removed by the RPG software, replace the ghost with -999.,
        # if noise factor 0 was selected in the RPG software, replace the ghost by the minimum noise value, to avoid
        # wrong noise estimations (to much signal would be lost otherwise),
        idx_ts_nf0 = np.argwhere(LIMRAD_Zspec[2]['var'][:, 0, 0] != -999.0)
    #
        if idx_ts_nf0.size > 0:
            mask_left, mask_right = Ze_lin_left < c3_Ze_max, Ze_lin_right < c3_Ze_max
            min_left, min_right = np.amin(Ze_lin_left, axis=2), np.amin(Ze_lin_right, axis=2)
    #
            for i_bin in range(mask_left.shape[2]):
                Ze_lin_left[mask_left[:, :, i_bin], i_bin] = min_left[mask_left[:, :, i_bin]]
            for i_bin in range(mask_right.shape[2]):
                Ze_lin_right[mask_right[:, :, i_bin], i_bin] = min_right[mask_right[:, :, i_bin]]
        else:
            Ze_lin_left[Ze_lin_left < c3_Ze_max] = -999.0
            Ze_lin_right[Ze_lin_right < c3_Ze_max] = -999.0
    #
        LIMRAD_Zspec[2]['var'][:, :, :idx_left] = Ze_lin_left.copy()
        LIMRAD_Zspec[2]['var'][:, :, idx_right:] = Ze_lin_right.copy()
        print('filtered ghost echos in chirp 3, elapsed time = {:.3f} sec.'.format(time.time() - tstart))
    #
    ######################################################################
    #
    # noise estimation a la Hildebrand & Sekhon
    # Logicals for different tasks
    include_noise = True if nsd < 0.0 else False
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True

    # remove noise from raw spectra and calculate radar moments
    # dimensions:
    #       -   noise_est[:]['signal']      [Nchirps][ntime, nrange]
    #       -   noise_est[:]['threshold']   [Nchirps][ntime, nrange]
    noise_est = s2m.noise_estimation(LIMRAD_Zspec, n_std_deviations=nsd,
                                     include_noise=include_noise,
                                     main_peak=main_peak)

    ######################################################################
    #
    # moment calculation
    LIMRAD_moments = s2m.spectra_to_moments_rpgfmcw94(LIMRAD_Zspec, noise_est,
                                                      include_noise=include_noise,
                                                      main_peak=main_peak)
    invalid_mask = LIMRAD_moments['mask']

    ######################################################################
    #
    # 1st chirp ghost echo filter
    # test differential phase filter technique
    if 'filter_ghost_C1' in kwargs and kwargs['filter_ghost_C1']:
        tstart = time.time()

        # setting higher threshold if chirp 2 contains high reflectivity values
        sum_over_heightC1 = np.ma.sum(LIMRAD_moments['Ze'][:rg_offsets[1], :], axis=0)
        sum_over_heightC2 = np.ma.sum(LIMRAD_moments['Ze'][rg_offsets[1]:rg_offsets[2], :], axis=0)

        # load sensitivity limits (time, height) and calculate the mean over time
        sens_reduction = 15.0  # sensitivity in chirp 1 is reduced by 12.0 dBZ
        C2_Ze_threshold = 18.0  # if sum(C2_Ze) > this threshold, ghost echo in C1 is assumed
        LIMRAD94_SLv = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], [0, 'max'])
        sens_lim = h.z2lin(h.lin2z(np.mean(LIMRAD94_SLv['var'], axis=0)) + sens_reduction)[:rg_offsets[1]]
        ts_to_mask = np.argwhere(h.lin2z(sum_over_heightC2) > C2_Ze_threshold)[:, 0]

        m1 = invalid_mask[:rg_offsets[1], :].copy()
        for idx_ts in ts_to_mask:
            m1[:, idx_ts] = LIMRAD_moments['Ze'][:rg_offsets[1], idx_ts] < sens_lim

        invalid_mask[:rg_offsets[1], :] = m1.copy()

        for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
            LIMRAD_moments[mom][:rg_offsets[1], :] = np.ma.masked_where(m1, LIMRAD_moments[mom][:rg_offsets[1], :])

        print('filtered ghost echos in chirp 1, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    # despeckle the moments
    if 'despeckle' in kwargs and kwargs['despeckle']:
        tstart = time.time()
        # copy and convert from bool to 0 and 1, remove a pixelm if more than 20 neighbours are present (5x5 grid)
        new_mask = s2m.despeckle(invalid_mask.copy() * 1, 20)
        invalid_mask[new_mask == 1] = True

        for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
            LIMRAD_moments[mom] = np.ma.masked_where(new_mask == 1, LIMRAD_moments[mom])

        print('despeckle done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    ######################################################################
    #
    # build larda containers from calculated moments
    container_dict = {}
    for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        container_dict.update({mom: make_container_from_spectra(LIMRAD_Zspec, LIMRAD_moments[mom].T,
                                                                paraminfo[mom], invalid_mask.T)})

    return container_dict, LIMRAD_Zspec


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
    larda = pyLARDA.LARDA().connect('lacros_dacapo')
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
        date = '20190110'
        begin_dt = datetime.datetime.strptime(date + ' 22:00:05', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')

    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    LIMRAD94_moments, LIMRAD94_spectra = calculate_moments_from_spectra_rpgfmcw94(begin_dt, end_dt,
                                                                                  std_above_mean_noise,
                                                                                  larda.connectors[
                                                                                      'LIMRAD94'].system_info['params'],
                                                                                  despeckle=True,
                                                                                  filter_ghost_C1=True,
                                                                                  filter_ghost_C3=True,
                                                                                  main_peak=True)

    ########################################################################################################################
    #
    #   _ _ _ ____ _ ___ ____    ____ ____ _    _ ___  ____ ____ ___ ____ ___     _  _ ____    ____ _ _    ____
    #   | | | |__/ |  |  |___    |    |__| |    | |__] |__/ |__|  |  |___ |  \    |\ | |       |___ | |    |___
    #   |_|_| |  \ |  |  |___    |___ |  | |___ | |__] |  \ |  |  |  |___ |__/    | \| |___    |    | |___ |___
    #
    #
    #
    for var in ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP', 'MaxVel', 'C1Range', 'C2Range', 'C3Range']:
        print('loading variable from LV1 :: ' + var)
        LIMRAD94_moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    LIMRAD94_moments['DiffAtt']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                         LIMRAD94_moments['DiffAtt']['var'])
    LIMRAD94_moments['ldr']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                     LIMRAD94_moments['ldr']['var'])

    cloudnet_remsens_lim_path = '/lacroshome/remsens_lim/data/cloudnet/'

    if 'path' in kwargs:
        path = kwargs['path']
    else:
        if c_info[0] == 'Punta Arenas':
            path = cloudnet_remsens_lim_path + 'punta-arenas/' + 'calibrated/limrad94/' + date[:4] + '/'
        elif c_info[0] == 'Leipzig':
            path = cloudnet_remsens_lim_path + 'leipzig/' + 'calibrated/limrad94/' + date[:4] + '/'
        else:
            print('Error: No other sites implemented jet!')
            sys.exit(-42)

    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_moments, path)

    ########################################################################################################################

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))

