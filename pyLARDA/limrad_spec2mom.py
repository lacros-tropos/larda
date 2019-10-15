"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python limrad_spec2mom.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""

import sys, datetime, time

sys.path.append('../larda/')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpecToMom_Util as s2m

import logging

import numpy as np


def build_extended_container(larda, spectra_ch, begin_dt, end_dt, **kwargs):
    # read limrad94 doppler spectra and caluclate radar moments
    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    AvgNum_in = larda.read("LIMRAD94", "AvgNum", [begin_dt, end_dt])
    DoppLen_in = larda.read("LIMRAD94", "DoppLen", [begin_dt, end_dt])
    MaxVel_in = larda.read("LIMRAD94", "MaxVel", [begin_dt, end_dt])

    if spectra_ch[0] == 'H':
        SensitivityLimit = larda.read("LIMRAD94", "SLh", [begin_dt, end_dt], [0, 'max'])
    else:
        SensitivityLimit = larda.read("LIMRAD94", "SLv", [begin_dt, end_dt], [0, 'max'])


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
    Zspec = []
    for ic in range(n_chirps):
        tstart = time.time()
        Zspec.append(larda.read("LIMRAD94", f"C{ic+1}{spectra_ch}", [begin_dt, end_dt], [0, 'max']))
        ic_n_ts, ic_n_rg, ic_n_nfft = Zspec[ic]['var'].shape
        rg_offsets.append(rg_offsets[ic] + ic_n_rg)
        Zspec[ic].update({'no_av': np.divide(AvgNum[ic], DoppLen[ic]),
                          'DoppRes': DoppRes[ic],
                          'SL': SensitivityLimit['var'][:, rg_offsets[ic]:rg_offsets[ic+1]],
                          'NF': std_above_mean_noise})
        print(f'reading C{ic+1}{spectra_ch}, elapsed time = {time.time() - tstart:.3f} sec.')

    for ic in range(n_chirps):
        Zspec[ic]['rg_offsets'] = rg_offsets

    return Zspec

def calculate_moments_from_spectra_rpgfmcw94(Z_spec, paraminfo, **kwargs):
    """
    This routine calculates the radar moments: reflectivity, mean Doppler velocity, spectrum width, skewness and
    kurtosis from the level 0 spectrum files of the 94 GHz RPG cloud radar.

    Args:
        Z_spec (list of dicts): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        rg_offsets (list): containing indices of chrip change
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94
        SL (dict): larda container of sensitivity limit
        nsd (float): number of standard deviations above mean noise threshold

    Return:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """


    ####################################################################################################################
    #
    # 3rd chirp ghost echo filter
    if 'filter_ghost_C3' in kwargs and kwargs['filter_ghost_C3']:
        s2m.filter_ghost_echos_RPG94GHz_FMCW(Z_spec, C2C3=True)
    #
    ####################################################################################################################
    #
    # noise estimation a la Hildebrand & Sekhon
    # Logicals for different tasks
    nsd = Z_spec[0]['NF']
    include_noise = True if nsd < 0.0 else False
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True

    # remove noise from raw spectra and calculate radar moments
    # dimensions:
    #       -   noise_est[:]['signal']      [Nchirps][ntime, nrange]
    #       -   noise_est[:]['threshold']   [Nchirps][ntime, nrange]
    noise_est = s2m.noise_estimation(Z_spec, n_std_deviations=nsd,
                                     include_noise=include_noise,
                                     main_peak=main_peak)

    ####################################################################################################################
    #
    # moment calculation
    moments = s2m.spectra_to_moments_rpgfmcw94(Z_spec, noise_est,
                                                      include_noise=include_noise,
                                                      main_peak=main_peak)
    invalid_mask = moments['mask'].copy()

    # save noise estimation (mean noise, noise threshold to spectra dict
    for iC in range(len(Z_spec)):
        for key in noise_est[iC].keys():
            Z_spec[iC].update({key: noise_est[iC][key].copy()})

    ####################################################################################################################
    #
    # 1st chirp ghost echo filter
    # test differential phase filter technique

    if 'filter_ghost_C1' in kwargs and kwargs['filter_ghost_C1']:
        invalid_mask = s2m.filter_ghost_echos_RPG94GHz_FMCW(moments,
                                                            C1=True,
                                                            inv_mask=invalid_mask,
                                                            offset=Z_spec[0]['rg_offsets'],
                                                            SL=Z_spec[0]['SL'])

    ####################################################################################################################
    #
    # despeckle the moments
    if 'despeckle' in kwargs and kwargs['despeckle']:
        tstart = time.time()
        # copy and convert from bool to 0 and 1, remove a pixel  if more than 20 neighbours are invalid (5x5 grid)
        new_mask = s2m.despeckle(invalid_mask.copy() * 1, 20)
        invalid_mask[new_mask == 1] = True

        # for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        #    moments[mom] = np.ma.masked_where(new_mask == 1, moments[mom])

        print('despeckle done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    ####################################################################################################################
    #
    # build larda containers from calculated moments
    container_dict = {}
    for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        print(mom)
        container_dict.update({mom: s2m.make_container_from_spectra(Z_spec, moments[mom].T,
                                                                    paraminfo[mom], invalid_mask.T)})

    return container_dict, Z_spec


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
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    if 'date' in kwargs:
        date = str(kwargs['date'])
        begin_dt = datetime.datetime.strptime(date + ' 00:00:10', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:50', '%Y%m%d %H:%M:%S')
    else:
        # date = '2019050'
        # begin_dt = datetime.datetime.strptime(date + ' 05:59:55', '%Y%m%d %H:%M:%S')
        # end_dt = datetime.datetime.strptime(date + ' 10:00:05', '%Y%m%d %H:%M:%S')

        date = '2019050'

        #begin_dt = datetime.datetime(2019, 5, 4, 8, 59, 55)
        #end_dt = datetime.datetime(2019, 5, 4, 12, 30, 5)

        #begin_dt = datetime.datetime(2019, 5, 4, 10, 36, 14)
        #end_dt = datetime.datetime(2019, 5, 4, 10, 36, 16)
        # begin_dt = datetime.datetime(2019, 5, 4, 10, 44, 55)
        # end_dt = datetime.datetime(2019, 5, 4, 10, 45, 6)

        #begin_dt = datetime.datetime(2019, 3, 13, 8, 59, 55)
        #end_dt = datetime.datetime(2019, 3, 13, 15, 30, 5)

        #begin_dt = datetime.datetime(2019, 8, 1, 6, 44, 16)
        #end_dt = datetime.datetime(2019, 8, 1, 6, 44, 20)

        #begin_dt = datetime.datetime(2019, 3, 15, 11, 29, 55)
        #end_dt = datetime.datetime(2019, 3, 15, 14, 30, 5)

        begin_dt = datetime.datetime(2019, 9, 8, 0, 0, 5)
        end_dt = datetime.datetime(2019, 9, 8, 23, 59, 55)

    std_above_mean_noise = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    LIMRAD_Zspec = build_extended_container(larda, 'VSpec', begin_dt, end_dt)

    LIMRAD94_moments, LIMRAD94_spectra = calculate_moments_from_spectra_rpgfmcw94(LIMRAD_Zspec,
                                                                                  larda.connectors[
                                                                                      'LIMRAD94'].system_info['params'],
                                                                                  nsd=std_above_mean_noise,
                                                                                  despeckle=True,
                                                                                  filter_ghost_C1=True,
                                                                                  filter_ghost_C3=True,
                                                                                  main_peak=True)

    for var in ['ldr', 'rr', 'LWP', 'SurfTemp', 'SurfWS']:
        print('loading variable from LV1 :: ' + var)
        LIMRAD94_moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    #
    LIMRAD94_moments['ldr']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
                                                        LIMRAD94_moments['ldr']['var'])

    #    MIRA_moments = {}
    #    LIMRAD_vars = ['Ze', 'VEL', 'sw', 'ldr', 'rr', 'LWP', 'SurfTemp', 'SurfWS']
    #
    #    cnt = 0
    #    for var in ['Zg', 'VELg', 'sw', 'LDRg']:
    #        print('loading variable from LV1 :: ' + var)
    #        MIRA_moments.update({LIMRAD_vars[cnt]: larda.read("MIRA", var, [begin_dt, end_dt], [0, 'max'])})
    #        cnt += 1
    #
    #    for var in ['rr', 'LWP', 'SurfTemp', 'SurfWS']:
    #        print('loading variable from LV1 :: ' + var)
    #        MIRA_moments.update({LIMRAD_vars[cnt]: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    #        cnt += 1
    #

    # loading cloudnet temperature data
    T = larda.read("CLOUDNET_LIMRAD", "T", [begin_dt, end_dt], [0, 'max'])


    def toC(datalist):
        return datalist[0]['var'] - 273.15, datalist[0]['mask']


    T = pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"})
    contour = {'data': T, 'levels': np.arange(-35, 1, 10)}

    ########################################################################################################FONT=CYBERMEDIUM
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    #
    #
    plot_remsen_ql = True
    plot_radar_moments = False
    plot_range = [0, 6000]
    fig_size = [6, 5]
    import os

    if plot_radar_moments:
        # create folder for subfolders if it doesn't exist already
        if not os.path.isdir('first_talk'): os.mkdir('first_talk')
        os.chdir('first_talk')
        print('\ncd to :: ', 'first_talk')

        # filter_str = 'no'
        # filter_str = 'C1'
        filter_str = 'all'

        LIMRAD94_ZE = LIMRAD94_moments['Ze']
        LIMRAD94_ZE['var_unit'] = 'dBZ'
        LIMRAD94_ZE['colormap'] = 'jet'
        LIMRAD94_ZE['var_lims'] = [-50, 20]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE, fig_size=fig_size, range_interval=plot_range,
                                                         z_converter='lin2z',
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_Ze' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        LIMRAD94_VEL = LIMRAD94_moments['VEL']
        LIMRAD94_VEL['var_lims'] = [-4, 2]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, fig_size=fig_size, range_interval=plot_range,
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        LIMRAD94_sw = LIMRAD94_moments['sw']
        LIMRAD94_sw['var_lims'] = [0, 0.5]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_sw, fig_size=fig_size, range_interval=plot_range,
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        LIMRAD94_skew = LIMRAD94_moments['skew']
        LIMRAD94_skew['var_lims'] = [-0.5, 1]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_skew, fig_size=fig_size, range_interval=plot_range,
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_skew' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)
        

        LIMRAD94_skew_smoothed = LIMRAD94_skew
        LIMRAD94_skew_smoothed['var_lims'] = [-0.5, 1]

        import scipy as sp
        import scipy.ndimage

        sigma_y = 1.0
        sigma_x = 1.0
        sigma = [sigma_y, sigma_x]

        LIMRAD94_skew_smoothed['var'] = sp.ndimage.filters.gaussian_filter(LIMRAD94_skew['var'].copy(),
                                                                           sigma=1, mode='nearest')


        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_skew_smoothed, fig_size=fig_size, range_interval=plot_range,
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_skew_smoothed' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)


        LIMRAD94_kurt = LIMRAD94_moments['kurt']
        LIMRAD94_kurt['var_lims'] = [1, 6]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_kurt, fig_size=fig_size, range_interval=plot_range,
                                                         #contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_kurt' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}_{filter_str}filter.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        #
        ##    LIMRAD94_VEL = LIMRAD94_moments['VEL']
        ##    LIMRAD94_VEL['var_lims'] = [-4, 2]
        ##    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, fig_size=[16, 3], range_interval=[0, 12000])
        ##    fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:o%H%M%S_}' + '0-12000' + '_filter.png'
        ##    fig.savefig(fig_name, dpi=250)
        #
        #    import scipy.ndimage as ndimage
        #
        #    LIMRAD94_ZE['var'] = ndimage.gaussian_filter(LIMRAD94_ZE['var'], sigma=1.0, order=0)
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE, fig_size=[16, 3], range_interval=[0, 12000],
        #                                                     #z_converter='lin2z',
        #                                                     title='filtered, smothed')
        #    fig_name = 'limrad_skew' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '_filter-smothed.png'
        #    fig.savefig(fig_name, dpi=250)
        #
        #    LIMRAD94_sw = LIMRAD94_moments['sw']
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_sw, fig_size=[16, 3], range_interval=[0, 12000])
        #    fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '_filter..png'
        #    fig.savefig(fig_name, dpi=250)
    ########################################################################################################################
    plot_single_spectra = False
    plot_time_spectro = False
    plot_range_spectro = False

    use_interp_specctra = False

    ########################################################################################################################
    #
    #  ___  _    ____ ___    ____ _ _  _ ____ _    ____    ____ ___  ____ ____ ___ ____ ____ ____
    #  |__] |    |  |  |     [__  | |\ | | __ |    |___    [__  |__] |___ |     |  |__/ |__| [__
    #  |    |___ |__|  |     ___] | | \| |__] |___ |___    ___] |    |___ |___  |  |  \ |  | ___]

    #

    if plot_single_spectra:
        name = os.path.dirname(__file__) + f'/spectra_png/spectra_{begin_dt:%Y%m%d_%H%M%S}/'

        # create folder for subfolders if it doesn't exist already
        import os, time

        if not os.path.isdir(name): os.mkdir(name)
        os.chdir(name)
        print('\ncd to :: ', name)

        MIRA_Zspec = larda.read("MIRA", "Zspec", [begin_dt], [0, 'max'])

        for ic in range(len(LIMRAD94_spectra)-1):
            vel_min, vel_max = LIMRAD94_spectra[ic+1]['vel'].min(), LIMRAD94_spectra[ic+1]['vel'].max()
            for rg in LIMRAD94_spectra[ic]['rg']:
                if plot_range[0] < rg < plot_range[1]:

                    fig, ax = pyLARDA.Transformations.plot_spectra(LIMRAD94_spectra[ic],
                                                                   MIRA_Zspec,
                                                                   #mean=LIMRAD94_spectra[ic]['mean'],
                                                                   #thresh=LIMRAD94_spectra[ic]['threshold'],
                                                                   vmin=-50, vmax=10, velmin=-4, velmax=2,
                                                                   z_converter='lin2z', save=name+ f'{ic}',
                                                                   text=False, fig_size=fig_size
                                                                   )

                    break


    if plot_time_spectro:
        name = os.path.dirname(__file__) + f'/spectra_png/time_spectro_{begin_dt:%Y%m%d_%H%M%S}/'

        # create folder for subfolders if it doesn't exist already
        import os, time

        if not os.path.isdir(name): os.mkdir(name)
        os.chdir(name)
        print('\ncd to :: ', name)

        ts_list = LIMRAD94_spectra[0]['ts']

        cnt = 0
        for ic in range(len(LIMRAD94_spectra)):
            rg_list = LIMRAD94_spectra[ic]['rg']

            for iR, rg in zip(range(LIMRAD94_spectra[ic]['rg'].size), rg_list):
                if plot_range[0] < rg < plot_range[-1]:
                    intervall = {'time': [ts_list[0], ts_list[-1]], 'range': [rg]}
                    spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD94_spectra[ic], value=intervall)
                    for iT in range(spectrogram_LIMRAD['ts'].size):
                        thresh = LIMRAD94_spectra[ic]['threshold'][iT, iR]
                        spectrogram_LIMRAD['var'][iT, spectrogram_LIMRAD['var'][iT, :] < thresh] = -999.0

                    spectrogram_LIMRAD['var'] = np.ma.masked_less_equal(spectrogram_LIMRAD['var'], 0.0)

                    spectrogram_LIMRAD['colormap'] = 'jet'
                    spectrogram_LIMRAD['var_lims'] = [-50, 10]
                    vel_min, vel_max = spectrogram_LIMRAD['vel'].min(), spectrogram_LIMRAD['vel'].max()

                    fig, ax = pyLARDA.Transformations.plot_spectrogram(spectrogram_LIMRAD, fig_size=fig_size,
                                                                       z_converter='lin2z', v_lims=[vel_min, vel_max],
                                                                       grid='major')

                    fig_name = name + f'limrad_{str(cnt).zfill(4)}_{begin_dt:%Y%m%d}_time_spectrogram.png'
                    fig.savefig(fig_name, dpi=200)
                    print('figure saved :: ', fig_name)
                    cnt += 1

    #######################################################################################################
    #
    #   ______ _______ __   _  ______ _______      _______  _____  _______ _______ _______  ______  _____
    #  |_____/ |_____| | \  | |  ____ |______      |______ |_____] |______ |          |    |_____/ |     |
    #  |    \_ |     | |  \_| |_____| |______      ______| |       |______ |_____     |    |    \_ |_____|
    #

    if plot_range_spectro:
        name = os.path.dirname(__file__) + f'/spectra_png/range_spectro_{begin_dt:%Y%m%d_%H%M%S}/'
        #
        # create folder for subfolders if it doesn't exist already
        import os, time

        #
        if not os.path.isdir(name): os.mkdir(name)
        os.chdir(name)
        print('\ncd to :: ', name)
        #
        # LIMRAD_Zspec2 = larda.read("LIMRAD94", "VSpec", [begin_dt, end_dt], plot_range)
        ##MIRA_ZSpec = larda.read("MIRA", "Zspec", [begin_dt, end_dt], [0, 'max'])
        #
        ##shaun_wind_vec = larda.read("SHAUN", "wind_direction", [begin_dt, end_dt], [0, 'max'])
        ##fig, _ = pyLARDA.Transformations.plot_barbs_timeheight(shaun_wind_vec, fig_size=[8, 6], range_interval=[0, 'max'])
        ##fig_name = 'wind_direction_' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}.png'
        ##fig.tight_layout(rect=[0, 0, 1, 0.95])
        ##fig.savefig(fig_name, dpi=250)
        #
        ts_list = LIMRAD94_spectra[0]['ts']
        ##ts_list = MIRA_ZSpec['ts']


        for ic in range(len(LIMRAD94_spectra)):
            for iT in range(LIMRAD94_spectra[ic]['ts'].size):
                for iR in range(LIMRAD94_spectra[ic]['rg'].size):
                    thresh = LIMRAD94_spectra[ic]['threshold'][iT, iR]
                    LIMRAD94_spectra[ic]['var'][iT, iR, LIMRAD94_spectra[ic]['var'][iT, iR, :] < thresh] = -999.0
                    #
            LIMRAD94_spectra[ic]['var'] = np.ma.masked_less_equal(LIMRAD94_spectra[ic]['var'], 0.0)


        if plot_range[1] <= LIMRAD94_spectra[2]['rg'].max()+100.:
            ic = 2
        if plot_range[1] <= LIMRAD94_spectra[1]['rg'].max()+100.:
            ic = 1
        if plot_range[1] <= LIMRAD94_spectra[0]['rg'].max()+100.:
            ic = 0

        cnt = 0
        if use_interp_specctra:
            LIMRAD94_spectra_interp = larda.read("LIMRAD94", "VSpec", [begin_dt, end_dt], [0, 'max'])
            n_ts = LIMRAD94_spectra_interp['ts'].size
        else:
            n_ts = LIMRAD94_spectra[ic]['ts'].size

        for iT, ts in zip(range(n_ts), ts_list):
            intervall = {'time': [ts], 'range': plot_range}
            if use_interp_specctra:
                spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD94_spectra_interp, value=intervall)
            else:
                spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(LIMRAD94_spectra[ic], value=intervall)
            #

            spectrogram_LIMRAD['colormap'] = 'jet'
            spectrogram_LIMRAD['var_lims'] = [-45, -5]
            #
            #
            fig, ax = pyLARDA.Transformations.plot_spectrogram(spectrogram_LIMRAD, fig_size=fig_size,
                                                               z_converter='lin2z', v_lims=[-5, 5],
                                                               grid='major')
            #
            fig_name = name + f'limrad_{str(cnt).zfill(4)}_{begin_dt:%Y%m%d}_range_spectrogram.png'
            fig.savefig(fig_name, dpi=200)
            print('figure saved :: ', fig_name)
            cnt += 1

        #
        #
        #   spectrogram_MIRA   = pyLARDA.Transformations.slice_container(MIRA_ZSpec, value=intervall)
        ##
        #   spectrogram_MIRA['colormap'] = 'jet'
        #   spectrogram_MIRA['var_lims'] = [-50, 0]
        ##
        #   fig, ax = pyLARDA.Transformations.plot_spectrogram(spectrogram_MIRA, fig_size=[6, 4],
        #                                                       z_converter='lin2z', v_lims=[-5, 5], grid='major')
        ##
        #   fig_name = name + f'mira_{str(cnt).zfill(4)}_{begin_dt:%Y%m%d}_range_spectrogram.png'
        #   fig.savefig(fig_name, dpi=200)
        #   print('figure saved :: ', fig_name)

    ########################################################################################################################
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ _  _ ____ ____ _  _ ____    ____ _
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |___ |\/| [__  |___ |\ | [__     |  | |
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |___ |  | ___] |___ | \| ___]    |_\| |___
    #
    #
    #
    if plot_remsen_ql:

       fig, ax = pyLARDA.Transformations.remsens_limrad_quicklooks(LIMRAD94_moments)
       fig_name = f'{begin_dt:%Y%m%d}_QL_LIMRAD94.png'

       #fig, ax = pyLARDA.Transformations.remsens_limrad_quicklooks(MIRA_moments)
       #fig_name = f'{begin_dt:%Y%m%d}_QL_MIRA35.png'

       fig.savefig(fig_name, dpi=300)
########################################################################################################################

########################################################################################################################
#
#   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ _  _ ____ _ ___ _ _  _ _ ___ _   _    _    _ _  _ _ ___ ____
#   |__] |    |  |  |   |  | |\ | | __    [__  |___ |\ | [__  |  |  | |  | |  |   \_/     |    | |\/| |  |  [__
#   |    |___ |__|  |   |  | | \| |__]    ___] |___ | \| ___] |  |  |  \/  |  |    |      |___ | |  | |  |  ___]
#
#
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
#   _ _ _ ____ _ ___ ____    ____ ____ _    _ ___  ____ ____ ___ ____ ___     _  _ ____    ____ _ _    ____
#   | | | |__/ |  |  |___    |    |__| |    | |__] |__/ |__|  |  |___ |  \    |\ | |       |___ | |    |___
#   |_|_| |  \ |  |  |___    |___ |  | |___ | |__] |  \ |  |  |  |___ |__/    | \| |___    |    | |___ |___
#
#
#
#    import pyLARDA.NcWrite as nc
#    for var in ['DiffAtt', 'ldr', 'bt', 'rr', 'LWP', 'MaxVel', 'C1Range', 'C2Range', 'C3Range']:
#        print('loading variable from LV1 :: ' + var)
#        LIMRAD94_moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
#    #
#    LIMRAD94_moments['DiffAtt']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
#                                                         LIMRAD94_moments['DiffAtt']['var'])
#    LIMRAD94_moments['ldr']['var'] = np.ma.masked_where(LIMRAD94_moments['Ze']['mask'] == True,
#                                                     LIMRAD94_moments['ldr']['var'])
#    #
#    cloudnet_remsens_lim_path = '/media/sdig/LACROS/cloudnet/data/'
#    #
#    if 'path' in kwargs:
#        path = kwargs['path']
#    else:
#        if c_info[0] == 'Punta Arenas':
#            path = cloudnet_remsens_lim_path + 'punta-arenas/' + 'calibrated/limrad94/' + date[:4] + '/'
#        elif c_info[0] == 'Leipzig':
#            path = cloudnet_remsens_lim_path + 'leipzig/' + 'calibrated/limrad94/' + date[:4] + '/'
#        else:
#            print('Error: No other sites implemented jet!')
#            sys.exit(-42)
#    #
#    flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_moments, path)
#    #
#    ########################################################################################################################
#
#    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
