"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:

    .. code::

        python spec2mom_limrad94.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""
import bisect
import copy
import warnings

import datetime
import logging
import numpy as np
import pandas as pd
import sys
import time
from itertools import product
from numba import jit
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import correlate

from typing import List, Set, Dict, Tuple, Optional, Union

warnings.simplefilter("ignore", UserWarning)
sys.path.append('../../larda/')

from pyLARDA.helpers import z2lin, argnearest, lin2z, ts_to_dt, dt_to_ts

logger = logging.getLogger(__name__)


def replace_fill_value(data, newfill):
    """
    Replaces the fill value of an spectrum container by their time and range specific mean noise level.

    Args:
        data (numpy.array) : 3D spectrum array (time, range, velocity)
        newfill (numpy.array) : 2D new fill values for 3rd dimension (velocity)

    Returns:
        var (numpy.array): spectrum with mean noise
    """

    n_ts, n_rg, _ = data.shape
    var = data.copy()
    masked = np.all(data <= 0.0, axis=2)

    for iT in range(n_ts):
        for iR in range(n_rg):
            if masked[iT, iR]:
                var[iT, iR, :] = newfill[iT, iR]
            else:
                var[iT, iR, var[iT, iR, :] <= 0.0] = newfill[iT, iR]
    return var


def get_chirp_from_range(rg_offsets, i_rg):
    for i, ioff in enumerate(rg_offsets[1:]):
        if i_rg <= ioff: return i


@jit(nopython=True, fastmath=True)
def estimate_noise_hs74(spectrum, navg=1, std_div=6.0, nnoise_min=1):
    """REFERENCE TO ARM PYART GITHUB REPO: https://github.com/ARM-DOE/pyart/blob/master/pyart/util/hildebrand_sekhon.py

    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.

    Args:
        spectrum (array): Doppler spectrum in linear units.
        navg (int, optional): The number of spectral bins over which a moving average 
            has been taken. Corresponds to the **p** variable from equation 9 of the article. 
            The default value of 1 is appropriate when no moving average has been applied to the spectrum.
        std_div (float, optional): Number of standard deviations above mean noise floor to specify the
            signal threshold, default: threshold=mean_noise + 6*std(mean_noise)
        nnoise_min (int, optional): Minimum number of noise samples to consider the estimation valid.

    Returns:
        tuple with

        - **mean** (*float*): Mean of points in the spectrum identified as noise.
        - **threshold** (*float*): Threshold separating noise from signal. The point in the spectrum with
          this value or below should be considered as noise, above this value
          signal. It is possible that all points in the spectrum are identified
          as noise. If a peak is required for moment calculation then the point
          with this value should be considered as signal.
        - **var** (*float*): Variance of the points in the spectrum identified as noise.
        - **nnoise** (*int*): Number of noise points in the spectrum.

    References:
        P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
        Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13, 808-811.
    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise

    rtest = 1 + 1 / navg
    sum1 = 0.
    sum2 = 0.
    for i, pwr in enumerate(sorted_spectrum):
        npts = i + 1
        if npts < nnoise_min:
            continue

        sum1 += pwr
        sum2 += pwr * pwr

        if npts * sum2 < sum1 * sum1 * rtest:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise.
            sum1 -= pwr
            sum2 -= pwr * pwr
            break

    mean = sum1 / nnoise
    var = sum2 / nnoise - mean * mean

    threshold = mean + np.sqrt(var) * std_div

    return mean, threshold, var, nnoise


@jit(nopython=True, fastmath=True)
def find_peak_edges(signal, threshold=-1, imaxima=-1):
    """Returns the indices of left and right edge of the main signal peak in a Doppler spectra.

    Args:
        signal (numpy.array): 1D array Doppler spectra
        threshold: noise threshold

    Returns:
        [index_left, index_right] (list): indices of signal minimum/maximum velocity
    """
    len_sig = len(signal)
    index_left, index_right = 0, len_sig
    if threshold < 0: threshold = np.min(signal)
    if imaxima < 0: imaxima = np.argmax(signal)

    for ispec in range(imaxima, len_sig):
        if signal[ispec] > threshold: continue
        index_right = ispec
        break

    for ispec in range(imaxima, -1, -1):
        if signal[ispec] > threshold: continue
        index_left = ispec + 1  # the +1 is important, otherwise a fill_value will corrupt the numba code
        break

    return threshold, [index_left, index_right]


@jit(nopython=True, fastmath=True)
def radar_moment_calculation(signal, vel_bins, DoppRes):
    """
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width,
    skewness, and kurtosis of one Doppler spectrum. Optimized for the use of Numba.

    Note:
        Divide the signal_sum by 2 because vertical and horizontal channel are added.
        Subtract half of of the Doppler resolution from mean Doppler velocity, because

    Args:
        - signal (float array): detected signal from a Doppler spectrum
        - vel_bins (float array): extracted velocity bins of the signal (same length as signal)
        - DoppRes (int): resolution of the Doppler spectra (different for each chirp)

    Returns:
        dict containing

        - **Ze_lin** (*float array*): reflectivity (0.Mom) over range of velocity bins [mm6/m3]
        - **VEL** (*float array*): mean velocity (1.Mom) over range of velocity bins [m/s]
        - **sw** (*float array*): spectrum width (2.Mom) over range of velocity bins [m/s]
        - **skew** (*float array*): skewness (3.Mom) over range of velocity bins
        - **kurt** (*float array*): kurtosis (4.Mom) over range of velocity bins
    """

    signal_sum = np.sum(signal)  # linear full spectrum Ze [mm^6/m^3], scalar
    Ze_lin = signal_sum / 2.0
    pwr_nrm = signal / signal_sum  # determine normalized power (NOT normalized by Vdop bins)

    VEL = np.sum(vel_bins * pwr_nrm)
    vel_diff = vel_bins - VEL
    vel_diff2 = vel_diff * vel_diff
    sw = np.sqrt(np.abs(np.sum(pwr_nrm * vel_diff2)))
    sw2 = sw * sw
    skew = np.sum(pwr_nrm * vel_diff * vel_diff2 / (sw * sw2))
    kurt = np.sum(pwr_nrm * vel_diff2 * vel_diff2 / (sw2 * sw2))
    VEL = VEL - DoppRes / 2.0

    return Ze_lin, VEL, sw, skew, kurt


@jit(nopython=True, fastmath=True)
def despeckle(mask, min_percentage):
    """Remove small patches (speckle) from any given mask by checking 5x5 box
    around each pixel, more than half of the points in the box need to be 1
    to keep the 1 at current pixel

    Args:
        mask (numpy.array, integer): 2D mask where 1 = an invalid/fill value and 0 = a data point (time, height)
        min_percentage (float): minimum percentage of neighbours that need to be signal above noise

    Returns:
        mask ... speckle-filtered matrix of 0 and 1 that represents (cloud) mask [height x time]

    """

    WSIZE = 5  # 5x5 window
    n_bins = WSIZE * WSIZE
    min_bins = int(min_percentage / 100 * n_bins)
    shift = int(WSIZE / 2)
    n_ts, n_rg = mask.shape

    for iT in range(n_ts - WSIZE):
        for iR in range(n_rg - WSIZE):
            if mask[iT, iR] and np.sum(mask[iT:iT + WSIZE, iR:iR + WSIZE]) > min_bins:
                mask[iT + shift, iR + shift] = True

    return mask


def make_container_from_spectra(spectra_all_chirps, values, paraminfo, invalid_mask, varname=''):
    """
    This routine will generate a larda container from calculated moments from spectra.

    Args:
        spectra_all_chirps (list of dicts): dimension [nchirps], containing the spectrum
                                            values of the 94 GHz RPG cloud radar
        values (numpy array): dimension [nrange, ntimes], values of calculated moments
        paraminfo (dict): information from params_[campaign].toml for the specific variable

    Returns:
        container (dict): larda data container
    """

    if len(varname) > 0:
        spectra_all_chirps = [spectra_all_chirps[ic][varname] for ic in range(len(spectra_all_chirps))]

    spectra = spectra_all_chirps[0]
    #np.array([rg for ic in spectra_all_chirps for rg in ic['rg']])
    container = {'dimlabel': ['time', 'range'],
                 'filename': spectra['filename'] if 'filename' in spectra else '',
                 'paraminfo': copy.deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'], 'colormap': paraminfo['colormap'],
                 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'],
                 'system': paraminfo['system'], 'name': paraminfo['paramkey'],
                 'rg': spectra['rg'], 'ts': spectra['ts'],
                 'mask': invalid_mask, 'var': values[:]}

    return container


def load_spectra_rpgfmcw94(larda, time_span, rpg_radar='LIMRAD94', **kwargs):
    """
    This routine will generate a list of larda containers including spectra of the RPG-FMCW 94GHz radar.
    The list-container at return will contain the additional information, for each chirp.

    Args:
        rpg_radar (string): name of the radar system as defined in the toml file
        larda (class larda): Initialized pyLARDA, already connected to a specific campaign
        time_span (list): Starting and ending time point in datetime format.
        **noise_factor (float): Noise factor, number of standard deviations from mean noise floor
        **ghost_echo_1 (bool): Filters ghost echos which occur over all chirps during precipitation.
        **ghost_echo_2 (bool): Filters ghost echos which occur over 1 chirps during precipitation.
        **estimate_noise (boal): If True, adds the following noise estimation values to the container:

            -   mean (2d ndarray): Mean noise level of the spectra.
            -   threshold (2d ndarray): Noise threshold, values above this threshold are consider as signal.
            -   variance (2d ndarray): The variance of the mean noise level.
            -   numnoise (2d ndarray): Number of Pixels that are cconsideras noise.
            -   signal (2d ndarray): Boolean array, a value is True if no signal was detected.
            -   bounds (3d ndarrax): Dimensions [n_time, n_range, 2] containing the integration boundaries.

    Returns:
        container (list): list of larda data container

        - **spec[i_chirps]['no_av']** (*float*): Number of spectral averages divided by the number of FFT points
        - **spec[i_chirps]['DoppRes']** (*float*): Doppler resolution for
        - **spec[i_chirps]['SL']** (*2D-float*): Sensitivity limit (dimensions: time, range)
        - **spec[i_chirps]['NF']** (*string*): Noise factor, default = 6.0
        - **spec[i_chirps]['rg_offsets']** (*list*): Indices, where chipr shifts
    """

    # read limrad94 doppler spectra and caluclate radar moments
    std_above_mean_noise = float(kwargs['noise_factor']) if 'noise_factor' in kwargs else 6.0
    heave_correct = kwargs['heave_correction'] if 'heave_correction' in kwargs else False
    version = kwargs['heave_corr_version'] if 'heave_corr_version' in kwargs else 'jr'
    add = kwargs['add'] if 'add' in kwargs else False
    shift = kwargs['shift'] if 'shift' in kwargs else 0
    dealiasing_flag = kwargs['dealiasing'] if 'dealiasing' in kwargs else False
    ghost_echo_1 = kwargs['ghost_echo_1'] if 'ghost_echo_1' in kwargs else True
    ghost_echo_2 = kwargs['ghost_echo_2'] if 'ghost_echo_2' in kwargs else True
    do_despeckle2D = kwargs['despeckle2D'] if 'despeckle2D' in kwargs else True
    add_horizontal_channel = True if 'add_horizontal_channel' in kwargs and kwargs['add_horizontal_channel'] else False
    estimate_noise = True if std_above_mean_noise > 0.0 else False

    AvgNum_in = larda.read(rpg_radar, "AvgNum", time_span)
    DoppLen_in = larda.read(rpg_radar, "DoppLen", time_span)
    MaxVel_in = larda.read(rpg_radar, "MaxVel", time_span)
    ChirpFFTSize_in = larda.read(rpg_radar, "ChirpFFTSize", time_span)
    SeqIntTime_in = larda.read(rpg_radar, "SeqIntTime", time_span)
    data = {}

    # depending on how much files are loaded, AvgNum and DoppLen are multidimensional list
    if len(AvgNum_in['var'].shape) > 1:
        AvgNum = AvgNum_in['var'][0]
        DoppLen = DoppLen_in['var'][0]
        ChirpFFTSize = ChirpFFTSize_in['var'][0]
        DoppRes = np.divide(2.0 * MaxVel_in['var'][0], DoppLen_in['var'][0])
        MaxVel = MaxVel_in['var'][0]
        SeqIntTime = SeqIntTime_in['var'][0]
    else:
        AvgNum = AvgNum_in['var']
        DoppLen = DoppLen_in['var']
        ChirpFFTSize = ChirpFFTSize_in['var']
        DoppRes = np.divide(2.0 * MaxVel_in['var'], DoppLen_in['var'])
        MaxVel = MaxVel_in['var']
        SeqIntTime = SeqIntTime_in['var']

    # initialize
    tstart = time.time()

    if add_horizontal_channel:
        data['SLh'] = larda.read(rpg_radar, "SLh", time_span, [0, 'max'])
        data['HSpec'] = larda.read(rpg_radar, 'HSpec', time_span, [0, 'max'])
        data['ReVHSpec'] = larda.read(rpg_radar, 'ImVHSpec', time_span, [0, 'max'])
        data['ImVHSpec'] = larda.read(rpg_radar, 'ReVHSpec', time_span, [0, 'max'])

    data['VHSpec'] = larda.read(rpg_radar, 'VSpec', time_span, [0, 'max'])
    data['SLv'] = larda.read(rpg_radar, "SLv", time_span, [0, 'max'])
    data['mdv'] = larda.read(rpg_radar, 'VEL', time_span, [0, 'max'])
    data['NF'] = std_above_mean_noise
    data['no_av'] = np.divide(AvgNum, DoppLen)
    data['DoppRes'] = DoppRes
    data['DoppLen'] = DoppLen
    data['MaxVel'] = MaxVel
    data['ChirpFFTSize'] = ChirpFFTSize
    data['SeqIntTime'] = SeqIntTime
    data['n_ts'], data['n_rg'], data['n_vel'] = data['VHSpec']['var'].shape
    data['n_ch'] = len(MaxVel)
    data['rg_offsets'] = [0]
    data['vel'] = []
    for var in ['C1Range', 'C2Range', 'C3Range']:
        logger.debug('loading variable from LV1 :: ' + var)
        data.update({var: larda.read(rpg_radar, var, time_span, [0, 'max'])})

    for ic in range(len(AvgNum)):
        nrange_ = larda.read(rpg_radar, f'C{ic + 1}Range', time_span)['var']
        if len(nrange_.shape) == 1:
            nrange_ = nrange_.size
        else:
            nrange_ = nrange_.shape[1]
        data['rg_offsets'].append(data['rg_offsets'][ic] + nrange_)
        data['vel'].append(np.linspace(-MaxVel[ic] + (0.5 * DoppRes[ic]), +MaxVel[ic] - (0.5 * DoppRes[ic]), np.max(DoppLen)))

    data['VHSpec']['rg_offsets'] = data['rg_offsets']

    logger.info(f'Loading spectra, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')
    """
    ####################################################################################################################
    ____ ___  ___  _ ___ _ ____ _  _ ____ _       ___  ____ ____ ___  ____ ____ ____ ____ ____ ____ _ _  _ ____ 
    |__| |  \ |  \ |  |  | |  | |\ | |__| |       |__] |__/ |___ |__] |__/ |  | |    |___ [__  [__  | |\ | | __ 
    |  | |__/ |__/ |  |  | |__| | \| |  | |___    |    |  \ |___ |    |  \ |__| |___ |___ ___] ___] | | \| |__]

    ####################################################################################################################                                                                                                             
    """

    if heave_correct:
        tstart = time.time()
        current_day = ts_to_dt(data['VHSpec']['ts'][0])
        data['VHSpec']['var'], data['heave_cor'], data['heave_cor_bins'], _, data['time_shift_array'] = heave_correction_spectra(
            data,
            current_day,
            path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
            mean_hr=True,
            only_heave=False,
            use_cross_product=True,
            transform_to_earth=True,
            add=add,
            shift=shift,
            version=version)

        logger.info(f'Heave correction applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    if do_despeckle2D:
        tstart = time.time()
        data['dspkl_mask'] = despeckle2D(data['VHSpec']['var'])
        data['VHSpec']['var'][data['dspkl_mask']], data['VHSpec']['mask'][data['dspkl_mask']] = -999.0, True
        logger.info(f'Despeckle applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    # read spectra and other variables
    if estimate_noise:
        tstart = time.time()
        data['edges'] = np.full((data['n_ts'], data['n_rg'], 2), 0, dtype=int)
        try:
            data['Vnoise'] = larda.read(rpg_radar, 'VNoisePow', time_span, [0, 'max'])
            if add_horizontal_channel: data['Hnoise'] = larda.read(rpg_radar, 'HNoisePow', time_span, [0, 'max'])

            # initialize arrays
            data['mean'] = np.full((data['n_ts'], data['n_rg']), -999.0)
            data['variance'] = np.full((data['n_ts'], data['n_rg']), -999.0)
            tmp = data['VHSpec']['var'].copy()
            tmp[tmp <= 0.0] = np.nan

            # catch RuntimeWarning: All-NaN slice encountered
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data['thresh'] = np.nanmin(tmp, axis=2)
                data['var_max'] = np.nanmax(tmp, axis=2)

            # find all-noise-spectra (aka. fill_value)
            mask = np.all(data['VHSpec']['var'] == -999.0, axis=2)
            data['thresh'][mask] = data['Vnoise']['var'][mask]
            del tmp

        except KeyError:
            logger.info('KeyError: Noise Power variable not found, calculate noise level...')
            noise_est = noise_estimation_uncompressed_data(data['VHSpec'], no_av=data['no_av'], n_std=6.0, rg_offsets=data['rg_offsets'])
            mask = ~noise_est['signal']
            data['thresh'] = noise_est['threshold']
            data['VHSpec']['var'][mask] = -999.0

            # IGNORES: RuntimeWarning: invalid value encountered in less:
            #                          masking = data['VHSpec']['var'][iT, iR, :] < data['thresh'][iT, iR]
            with np.errstate(invalid='ignore'):
                for iT, iR in product(range(data['n_ts']), range(data['n_rg'])):
                    if mask[iT, iR]: continue
                    masking = data['VHSpec']['var'][iT, iR, :] < data['thresh'][iT, iR]
                    data['VHSpec']['var'][iT, iR, masking] = -999.0

        if dealiasing_flag:
            dealiased_spec, dealiased_mask, new_vel, new_bounds, _, _ = dealiasing(
                data['VHSpec']['var'],
                data['vel'],
                data['SLv']['var'],
                data['rg_offsets'],
                vel_offsets=kwargs['dealiasing_vel'] if 'dealiasing_vel' in kwargs else None,
                show_triple=False
            )

            data['VHSpec']['var'] = dealiased_spec
            data['VHSpec']['mask'] = dealiased_mask
            data['VHSpec']['vel'] = new_vel[0]  # copy to larda container
            data['vel'] = new_vel  # copy all veloctiys
            data['edges'] = new_bounds

        else:
            for iT, iR in product(range(data['n_ts']), range(data['n_rg'])):
                if mask[iT, iR]: continue
                _, data['edges'][iT, iR, :] = find_peak_edges(data['VHSpec']['var'][iT, iR, :], data['thresh'][iT, iR])

        logger.info(f'Loading Noise Level, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    if ghost_echo_1:
        tstart = time.time()
        data['ge1_mask'] = filter_ghost_1(data['VHSpec']['var'], data['VHSpec']['rg'], data['vel'], data['rg_offsets'])
        logger.info(f'Precipitation Ghost Filter applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')
        logger.info(f'Number of ghost pixel due to precipitation = {np.sum(data["ge1_mask"])}')
        data['VHSpec']['var'][data['ge1_mask']], data['VHSpec']['mask'][data['ge1_mask']] = -999.0, True

    if ghost_echo_2:
        data['ge2_mask'] = filter_ghost_2(data['VHSpec']['var'], data['VHSpec']['rg'], data['SLv']['var'], data['rg_offsets'][1])
        logger.info(f'Curtain-like Ghost Filter applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')
        logger.info(f'Number of curtain-like ghost pixel = {np.sum(data["ge2_mask"])}')
        data['VHSpec']['var'][data['ge2_mask']], data['VHSpec']['mask'][data['ge2_mask']] = -999.0, True

    if do_despeckle2D:
        tstart = time.time()
        data['dspkl_mask'] = despeckle2D(data['VHSpec']['var'])
        data['VHSpec']['var'][data['dspkl_mask']], data['VHSpec']['mask'][data['dspkl_mask']] = -999.0, True
        logger.info(f'Despeckle applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    return data


def dealiasing_check(masked3D):
    """
    Checks for folding.

    Args:
        masked3D (numpy.array): 3D (time, range, velocity)
        vel (list): contains 1D numpy.arrays for each chirp
        mean_noise (numpy.array):
        rg_offsets (numpy.array):

    Returns:
        alias_flag (numpy.array):

    """

    frac = 5
    alias_flag = np.full(masked3D.shape[:2], False)
    masked2D = masked3D.all(axis=2)
    Nfft = masked3D.shape[2]
    frac = np.ceil(Nfft / 100 * frac).astype(np.int32)

    for iT, iR in product(range(masked3D.shape[0]), range(masked3D.shape[1])):

        if masked2D[iT, iR]: continue  # no signal was recorded

        # check if aliasing occured by checking if more than 'frac' percent of the bins exceeded
        # mean noise level at one of the spectra
        n_start = np.sum(~masked3D[iT, iR, :frac])
        n_end = np.sum(~masked3D[iT, iR, Nfft-frac+1:Nfft+1])

        if n_start >= frac or n_end >= frac: alias_flag[iT, iR] = True  # then aliasing detected


    return alias_flag


def dealiasing(
        spectra: np.array,
        vel_bins_per_chirp: List[np.array],
        noisefloor: np.array,
        rg_offsets: List[int] = None,
        show_triple: bool = False,
        vel_offsets: List[int] = None,
        jump: int = None,
) -> Union[np.array, np.array, List[np.array], np.array, np.array, np.array]:
    """
        Peaks exceeding the maximum unambiguous Doppler velocity range of ± v_Nyq in [m s-1]
        appear at the next upper (lower) range gate at the other end of the velocity spectrum.
        The dealiasing method presented here aims to correct for this and is applied to every time step.

        Logging level INFO shows the de-aliasing progress bar.

    Args:
        spectra: dim = (n_time, n_range, n_velocity) in linear units!
        vel_bins_per_chirp: len = (n_chirp), each list element contains a numpy array of velocity bins
        noisefloor: dim = (n_time, n_range) in linear units!
        rg_offsets (optional): dim = (n_chirp + 1), starting with 0, range indices where chirp shift occurs
        show_triple (optional): if True, return dealiased spectra including the triplication
        vel_offsets (optional): velocity window around the main peak [x1, x2], x1 < 0, x2 > 0 ! in [m s-1], default [-6.0, +9.0]
        jump (optional): maximum number of Doppler bins a spectrum can change in two adjacent range bins

    Returns:
        tuple containing

        - **dealiased_spectra**: dim = (n_time, n_range, 3 * n_velocity), de-aliased Doppler spectrum
        - **dealiased_mask**: dim = (n_time, n_range, 3 * n_velocity), True if no signal
        - **velocity_new**: len = (n_chirp), each list element contains a numpy array of velocity bins for the respective chirp of ± 3*v_Nyq in [m s-1]
        - **signal_boundaries**: indices of left and right edge of a signal, [-1, -1] if no signal
        - **search_path**: indices of left and right edge of the search path, [-1, -1] if no signal
        - **idx_peak_matrix**: indices of the main peaks, [NDbins / 2] if no signal

    .. todo::

        - add time--height mask for dealiasing
        - search window relative to [m s-1]
        - abs(idx_new_peak - mean_idx_last_ts) > 120: --> relativ

    """
    (n_ts, n_rg, n_vel), n_ch = spectra.shape, len(rg_offsets) - 1
    n_vel_new = 3 * n_vel

    k = 2

    if jump is None:
        jump = n_vel // 2

    # triplicate velocity bins
    velocity_new = []
    for v in vel_bins_per_chirp:
        vel_range = v[-1] - v[0]
        velocity_new.append(np.linspace(v[0] - vel_range, v[-1] + vel_range, n_vel_new))

    # set (n_rg, 2) array containing velocity index offset ±velocty_jump_tolerance from maxima from last range gate
    _one_in_all = [-7.0, +7.0] if vel_offsets is None else vel_offsets

    velocty_jump_tolerance = np.array([_one_in_all for _ in range(n_ch)])  # ± [m s-1]

    rg_diffs = np.diff(rg_offsets)
    Dopp_res = np.array([vel_bins_per_chirp[ic][1] - vel_bins_per_chirp[ic][0] for ic in range(n_ch)])

    iDbinTol = [velocty_jump_tolerance[ires, :] // res for ires, res in enumerate(Dopp_res)]
    iDbinTol = np.concatenate([np.array([iDbinTol[ic]] * rg_diffs[ic]) for ic in range(n_ch)]).astype(np.int)

    # triplicate spectra
    Z_linear = np.concatenate([spectra for _ in range(3)], axis=2)

    # initialize arrays for dealiasing
    window_fcn = np.kaiser(n_vel_new, 4.0)
    signal_boundaries = np.zeros((n_ts, n_rg, 2), dtype=np.int)
    search_path = np.zeros((n_ts, n_rg, 2), dtype=np.int)
    dealiased_spectra = np.full(Z_linear.shape, -999.0, dtype=np.float32)
    dealiased_mask = np.full(Z_linear.shape, True, dtype=np.bool)
    idx_peak_matrix = np.full((n_ts, n_rg), n_vel_new // 2, dtype=np.int)
    all_clear = np.all(np.all(spectra <= 0.0, axis=2), axis=1)
    noise = np.copy(noisefloor)
    noise_mask = spectra.min(axis=2) > noise
    noise[noise_mask] = spectra.min(axis=2)[noise_mask]

    logger.debug(f'Doppler resolution per chirp : {Dopp_res}')
    logger.info(f'Doppler spectra de-aliasing....... ')
    for iT in range(n_ts) if logger.level > 20 else tqdm(range(n_ts), unit=' timesteps', total=n_ts):

        # entire profile is clear sky
        if all_clear[iT]:  continue

        # assume no dealiasing at upper most range gate
        idx_last_peak = n_vel_new // 2

        # Top-Down approach: check range gates below
        for iR in range(n_rg - 1, -1, -1):

            # the search window for the next peak maximum surrounds ± velocity_jump_tolerance [m s-1] around the last peak maximum
            search_window = range(max(idx_last_peak + iDbinTol[iR, 0], 0), min(idx_last_peak + iDbinTol[iR, 1], n_vel_new))
            Z_windowed = Z_linear[iT, iR, :] * np.roll(window_fcn, n_vel_new // 2 - idx_last_peak)
            Z_windowed = Z_windowed[search_window]  # Note: Think about index shift!
            idx_new_peak = np.argmax(Z_windowed) + search_window[0]

            # check if Doppler velocity jumps more than 120 bins from last _eak max to new(=one rg below) peak max
            mean_idx_last_ts = int(np.mean(idx_peak_matrix[max(0, iT - k):min(iT + 1, n_ts), max(0, iR - 1):min(iR + k, n_rg)]))
            if abs(idx_new_peak - mean_idx_last_ts) > jump:
                logger.debug(f'jump at iT={iT}   iR={iR}')
                idx_new_peak = mean_idx_last_ts
                search_window = range(max(idx_new_peak + iDbinTol[iR, 0], 0), min(idx_new_peak + iDbinTol[iR, 1], n_vel_new))

            search_path[iT, iR, :] = [search_window[0], search_window[-1]]  # for plotting

            if search_window[0] < idx_new_peak < search_window[-1]:
                # calc signal boundaries
                _, _bnd = find_peak_edges(Z_linear[iT, iR, :], threshold=noise[iT, iR], imaxima=idx_new_peak)

                # safety precautions, if idx-left-bound > idx-right-bound --> no signal
                if _bnd[0] == _bnd[1] + 1:
                    # probably clear sky
                    idx_peak_matrix[iT, iR] = idx_last_peak
                    signal_boundaries[iT, iR, :] = [-1, -1]
                else:
                    signal_boundaries[iT, iR, :] = _bnd
                    idx_peak_matrix[iT, iR] = idx_new_peak
                    idx_last_peak = idx_new_peak
                    # if show_triple == True, copy all signals including the triplication else copy only the main signal and not the triplication
                    _bnd_tmp = [None, None] if show_triple else _bnd
                    dealiased_spectra[iT, iR, _bnd_tmp[0]:_bnd_tmp[1]] = Z_linear[iT, iR, _bnd_tmp[0]:_bnd_tmp[1]]
                    dealiased_mask[iT, iR, _bnd_tmp[0]:_bnd_tmp[1]] = False

            else:
                # last peak stays the same, no integration boundaries
                signal_boundaries[iT, iR, :] = [-1, -1]
                idx_peak_matrix[iT, iR] = idx_last_peak

            logger.debug(f'signal boundaries(iR == {iR}) = {signal_boundaries[iT, iR, :]}     '
                         f'idx_peak_max  {idx_peak_matrix[iT, iR]},     '
                         f'min val = noise floor : {Z_linear[iT, iR, :].min():.7f}, {noise[iT, iR]:.7f}     ')

    # clean up signal boundaries
    signal_boundaries[(signal_boundaries <= 0) + (signal_boundaries >= n_vel_new)] = -1
    return dealiased_spectra, dealiased_mask, velocity_new, signal_boundaries, search_path, idx_peak_matrix


def noise_estimation_uncompressed_data(data, n_std=6.0, **kwargs):
    """
    Creates a dict containing the noise threshold, mean noise level,
    the variance of the noise, the number of noise values in the spectrum,
    and the boundaries of the main signal peak, if there is one

    Args:
        data (dict): data container, containing data['var'] of dimension (n_ts, n_range, n_Doppler_bins)
        **n_std_deviations (float): threshold = number of standard deviations
                                    above mean noise floor, default: threshold is the value of the first
                                    non-noise value

    Returns:
        dict with noise floor estimation for all time and range points
    """

    spectra3D = data['var'].copy()
    n_ts, n_rg, n_vel = spectra3D.shape
    if 'rg_offsets' in kwargs:
        rg_offsets = np.copy(kwargs['rg_offsets'])
        rg_offsets[0] = -1
        rg_offsets[-1] += 1
    else:
        rg_offsets = [-1, n_rg + 1]
    no_av = kwargs['no_av'] if 'no_av' in kwargs else [1]

    # fill values needs to be masked for noise removal otherwise wrong results
    spectra3D[spectra3D == -999.0] = np.nan

    # Estimate Noise Floor for all chirps, time stemps and range gates aka. for all pixels
    # Algorithm used: Hildebrand & Sekhon

    # allocate numpy arrays
    noise_est = {
        'mean': np.zeros((n_ts, n_rg), dtype=np.float32),
        'threshold': np.zeros((n_ts, n_rg), dtype=np.float32),
        'variance': np.zeros((n_ts, n_rg), dtype=np.float32),
        'numnoise': np.zeros((n_ts, n_rg), dtype=np.int32),
        'signal': np.full((n_ts, n_rg), fill_value=True),
    }

    # gather noise level etc. for all chirps, range gates and times
    logger.info(f'Noise estimation for uncompressed spectra....... ')
    noise_free = np.isnan(spectra3D).any(axis=2)
    iterator = product(range(n_ts), range(n_rg)) if logger.level > 20 else tqdm(product(range(n_ts), range(n_rg)), total=n_ts * n_rg, unit=' spectra')
    for iT, iR in iterator:
        if noise_free[iT, iR]: continue
        mean, thresh, var, nnoise = estimate_noise_hs74(
            spectra3D[iT, iR, :],
            navg=no_av[getnointerval(rg_offsets, iR) - 1],
            std_div=n_std
        )

        noise_est['mean'][iT, iR] = mean
        noise_est['variance'][iT, iR] = var
        noise_est['numnoise'][iT, iR] = nnoise
        noise_est['threshold'][iT, iR] = thresh
        noise_est['signal'][iT, iR] = nnoise < n_vel

    return noise_est


def mira_noise_calculation(radar_const, SNRCorFaCo, HSDco, noise_power_co, range_ka):
    """

    Args:
        radar_const:
        SNRCorFaCo:
        HSDco:
        noise_power_co:
        range_ka:

    Returns:
        noise level in linear units
    """
    noise_ka_lin = np.zeros(HSDco.shape)
    for iT in range(len(radar_const)):
        noise_ka_lin[iT, :] = radar_const[iT] * SNRCorFaCo[iT, :] * HSDco[iT, :] / noise_power_co[iT] * (range_ka / 5000.) ^ 2.
    return noise_ka_lin


def getnointerval(intervals, i):
    return bisect.bisect_left(intervals, i)


def seconds_to_fstring(time_diff):
    return datetime.datetime.fromtimestamp(time_diff).strftime("%M:%S")


def despeckle2D(data, min_perc=80.0):
    """This function is used to remove all spectral lines for one time-range-pixel if surrounding% of the sourounding pixels are fill_values.

    Args:
        data (numpy.array): cloud radar Doppler spectra, dimensions: (time, range, velocity), unit: [mm6 mm-3 m s-1]

    Keyword Args:
        min_perc (float): minimum percentage value of neighbouring pixel, that need to be above the noise threshold

    Returns:
        mask (numpy.array, bool): where True = fill_value, and False = signal, dimensions: (time, range, velocity)

    """
    # there must be high levels of reflection/scattering in this region to produce ghost echos
    mask_2D = despeckle(np.all(data <= 0.0, axis=2), min_perc)
    mask = data <= 0.0
    for iBin in range(data.shape[2]):
        mask[:, :, iBin] = mask_2D

    return mask


def filter_ghost_1(data, rg, vel, offset, dBZ_thresh=-20.0, reduce_by=1.5, **kwargs):
    """This function is used to remove certain spectral lines "speckle ghost echoes" from all chirps of RPG FMCW 94GHz cloud radar spectra.
    The speckle occur usually near the maximum unambiguous Doppler velocity.

    Args:
        data (numpy.array): cloud radar Doppler spectra, dimensions: (time, range, velocity), unit: [mm6 mm-3 m s-1]
        rg (numpy.array): range values, unit [m]
        vel (list of numpy.arrays): contains the Doppler velocity values for each chirp, dimension = n_chirps
        offset (list, integer): range indices where the chirp changes takes place, dimension = n_chirps + 1 (starting with 0)
        dBZ_thresh (float): values below will be considered as ghost echo
        reduce_by (float): reduce the maximum unambiguous Doppler velocity by this amount in [m s-1]
        **ignore_chirp1 (bool): Don't filter ghost echos of this type for first chirp (set to True if not given)
        **Z_thresh (float): Ze in dBZ to be exceeded in lowest 500 m range for filter to be activated

    Returns:
        mask (numpy.array, bool): where True = fill_value, and False = signal, dimensions: (time, range, velocity)

    """
    ignore_chirp1 = True if not 'ignore_chirp1' in kwargs else kwargs['ignore_chirp1']
    # there must be high levels of reflection/scattering in this region to produce ghost echos
    RG_MIN_, RG_MAX_ = 0.0, 500.0  # range interval
    mask = data <= 0.0
    reflectivity_thresh = 0.0 if not 'Z_thresh' in kwargs else kwargs['Z_thresh']

    # check the if high signal occurred in 0m - 500m altitude (indicator for speckle ghost echos above)
    dBZ_max = np.max(data[:, argnearest(rg, RG_MIN_):argnearest(rg, RG_MAX_), :], axis=2)
    ts_to_mask = np.any(dBZ_max >= z2lin(reflectivity_thresh), axis=1)

    signal_min = z2lin(dBZ_thresh)
    n_vel = data.shape[2]

    for iC in range(len(vel)):
        if iC < 1 and ignore_chirp1:
            continue  # exclude first chirp because ghost is hidden under real signal anyway
        idx_max_vel_new = argnearest(vel[iC], vel[iC][-1] - reduce_by)
        for iV in range(n_vel - idx_max_vel_new):
            mask[ts_to_mask, offset[iC]:offset[iC + 1], iV] = data[ts_to_mask, offset[iC]:offset[iC + 1], iV] < signal_min
        for iV in range(idx_max_vel_new, n_vel):
            mask[ts_to_mask, offset[iC]:offset[iC + 1], iV] = data[ts_to_mask, offset[iC]:offset[iC + 1], iV] < signal_min

    return mask


def filter_ghost_2(data, rg, SL, first_offset, dBZ_thresh=-5.0, reduce_by=10.0):
    """This function is used to remove curtain-like ghost echoes
    from the first chirp of RPG FMCW 94GHz cloud radar spectra.

    Args:
        data (numpy.array): cloud radar Doppler spectra, dimensions: (time, range, velocity), unit: [mm6 mm-3 m s-1]
        rg (numpy.array): range values, unit [m]
        SL (numpy.array): sensitivity limit, dimension: (time, range), unit: [mm6 mm-3]
        first_offset (integer): range index where the first chirp change takes place
        dBZ_thresh (float): minimum threshold in [dBZ], where ghost echos can be assumed
        reduce_by (float): reduce the sensitivity limit by this amount of [dBZ]

    Returns:
        mask (numpy.array, bool): where True = fill_value, and False = signal, dimensions: (time, range, velocity)

    """
    # there must be high levels of reflection/scattering in this region to produce ghost echos
    RG_MIN_, RG_MAX_ = 1500.0, 6000.0  # range interval
    mask = data <= 0.0

    # check the if high signal occurred in 1500m - 4500m altitude (indicator for curtain like ghost echo)
    dBZ_max = np.max(data[:, argnearest(rg, RG_MIN_):argnearest(rg, RG_MAX_), :], axis=2)
    ts_to_mask = np.any(dBZ_max >= z2lin(dBZ_thresh), axis=1)
    sens_lim = SL * reduce_by
    for iT, mask_iT in enumerate(ts_to_mask):
        if mask_iT:
            for iV in range(data.shape[2]):
                mask[iT, :first_offset, iV] = data[iT, :first_offset, iV] < sens_lim[iT, :first_offset]

    return mask


def split_by_compression_status(var, mask):
    indices = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    split_int = np.split(var, indices)
    return split_int[0::2] if mask[0] else split_int[1::2]


def spectra2moments(ZSpec, paraminfo, **kwargs):
    """
    This routine calculates the radar moments: reflectivity, mean Doppler velocity, spectrum width, skewness and
    kurtosis from the level 0 spectrum files of the 94 GHz RPG cloud radar.

    Args:
        ZSpec (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Returns:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

    # initialize variables:
    n_ts, n_rg, n_vel = ZSpec['VHSpec']['var'].shape
    n_chirps = ZSpec['n_ch']
    Z = np.full((n_ts, n_rg), np.nan)
    V = np.full((n_ts, n_rg), np.nan)
    SW = np.full((n_ts, n_rg), np.nan)
    SK = np.full((n_ts, n_rg), np.nan)
    K = np.full((n_ts, n_rg), np.nan)

    spec_lin = ZSpec['VHSpec']['var'].copy()
    mask = spec_lin <= 0.0
    spec_lin[mask] = 0.0

    # combine the mask for "contains signal" with "signal has more than 1 spectral line"
    mask1 = np.all(mask, axis=2)
    mask2 = ZSpec['edges'][:, :, 1] - ZSpec['edges'][:, :, 0] <= 0
    mask3 = ZSpec['edges'][:, :, 1] - ZSpec['edges'][:, :, 0] >= n_vel
    mask = mask1 * mask2 * mask3

    for iC in range(n_chirps):
        tstart = time.time()
        for iR in range(ZSpec['rg_offsets'][iC], ZSpec['rg_offsets'][iC + 1]):  # range dimension
            for iT in range(n_ts):  # time dimension
                if mask[iT, iR]: continue
                lb, rb = ZSpec['edges'][iT, iR, :]
                Z[iT, iR], V[iT, iR], SW[iT, iR], SK[iT, iR], K[iT, iR] = \
                    radar_moment_calculation(spec_lin[iT, iR, lb:rb], ZSpec['vel'][iC][lb:rb], ZSpec['DoppRes'][iC])

        logger.info(f'Chirp {iC + 1} Moments Calculated, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    moments = {'Ze': Z, 'VEL': V, 'sw': SW, 'skew': SK, 'kurt': K}
    # create the mask where invalid values have been encountered
    invalid_mask = np.full((ZSpec['VHSpec']['var'].shape[:2]), True)
    invalid_mask[np.where(Z > 0.0)] = False

    # despeckle the moments
    if 'despeckle' in kwargs and kwargs['despeckle']:
        tstart = time.time()
        # copy and convert from bool to 0 and 1, remove a pixel  if more than 20 neighbours are invalid (5x5 grid)
        new_mask = despeckle(invalid_mask, 80.)
        invalid_mask[new_mask] = True
        logger.info(f'Despeckle done, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    # mask invalid values with fill_value = -999.0
    for mom in moments.keys():
        moments[mom][invalid_mask] = -999.0

    # build larda containers from calculated moments
    container_dict = {mom: make_container_from_spectra([ZSpec], moments[mom], paraminfo[mom], invalid_mask, 'VHSpec') for mom in moments.keys()}

    return container_dict


def heave_correction(moments, date, path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                     mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)
    Calculate heave rate from seapath measurements and create heave correction array. If Doppler velocity is given as an
    input, correct it and return an array with the corrected Doppler velocities.
    Without Doppler Velocity input, only the heave correction array is returned.

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py, C1/2/3_Range,
                 SeqIntTime and Inc_ElA (for time (ts)) from LV1 file
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as 
            described in https://repository.library.noaa.gov/view/noaa/17400
        add (bool): whether to add the heave rate or subtract it

    Returns: 
        A number of variables
        
        - **new_vel** (*ndarray*); corrected Doppler velocities, same shape as moments["VEL"]["var"] or list if no Doppler
          Velocity is given;
        - **heave_corr** (*ndarray*): heave rate closest to each radar timestep for each height bin, same shape as
          moments["VEL"]["var"];
        - **seapath_out** (*pd.DataFrame*): data frame with all heave information from the closest time steps to the chirps

    """
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    logger.info(f"Starting heave correction for {date:%Y-%m-%d}")
    seapath = read_seapath(date, path_to_seapath)

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    seapath = calc_heave_rate(seapath, only_heave=only_heave, use_cross_product=use_cross_product,
                              transform_to_earth=transform_to_earth)

    ####################################################################################################################
    # Calculating heave correction array and add to Doppler velocity
    ####################################################################################################################
    # make input container to calc_heave_corr function
    container = {'C1Range': moments['C1Range'], 'C2Range': moments['C2Range'], 'C3Range': moments['C3Range'],
                 'SeqIntTime': moments['SeqIntTime'], 'ts': moments['Inc_ElA']['ts']}
    heave_corr, seapath_out = calc_heave_corr(container, date, seapath, mean_hr=mean_hr)

    try:
        if add:
            # create new Doppler velocity by adding the heave rate of the closest time step
            new_vel = moments['VEL']['var'] + heave_corr
        elif not add:
            # create new Doppler velocity by subtracting the heave rate of the closest time step
            new_vel = moments['VEL']['var'] - heave_corr
        # set masked values back to -999 because they also get corrected
        new_vel[moments['VEL']['mask']] = -999
        logger.info(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_vel, heave_corr, seapath_out
    except KeyError:
        logger.info(f"No input Velocities found! Cannot correct Doppler Velocity.\n Returning only heave_corr array!")
        logger.info(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
        new_vel = ["I'm an empty list!"]  # create an empty list to return the same number of variables
        return new_vel, heave_corr, seapath_out


def heave_correction_spectra(data, date,
                             path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                             mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False,
                             **kwargs):
    """Shift Doppler spectra to correct for heave motion of ship (RV-Meteor)
    Calculate heave rate from seapath measurements and create heave correction array. Translate the heave correction to
    a number spectra bins by which to move each spectra. If Spectra are given, shift them and return a 3D array with the
    shifted spectra.
    Without spectra input, only the heave correction array and the array with the number if bins to move is returned.

    Args:
        data: LIMRAD94 data container filled with spectra and C1/2/3_Range, SeqIntTime, MaxVel, DoppLen from LV1 file;
            for Claudia's version the mean Doppler velocity is also needed
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as described in https://repository.library.noaa.gov/view/noaa/17400
        add (bool): whether to add the heave rate or subtract it
        **kwargs:
            shift (int): number of time steps to shift seapath data
            version (str): which version to use, 'ca' or 'jr'

    Returns: 
        A number of variables
        
        - **new_spectra** (*ndarray*); corrected Doppler velocities, same shape as data["VHSpec"]["var"] or list if no Doppler
          Spectra are given;
        - **heave_corr** (*ndarray*): heave rate closest to each radar timestep for each height bin, shape = (time x range);
        - **seapath_out** (*pd.DataFrame*): data frame with all heave information from the closest time steps to the chirps

    """
    # unpack kwargs
    version = kwargs['version'] if 'version' in kwargs else 'jr'
    shift = kwargs['shift'] if 'shift' in kwargs else 0
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    logger.info(f"Starting heave correction for {date:%Y-%m-%d}")
    if version == 'ca':
        seapath = read_seapath(date, path_to_seapath, output_format='xarray')
        seapath = f_shiftTimeDataset(seapath)
    elif version == 'jr':
        seapath = read_seapath(date, path_to_seapath)

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    if version == 'ca':
        seapath = calc_heave_rate_claudia(seapath)
    elif version == 'jr':
        seapath = calc_heave_rate(seapath, only_heave=only_heave, use_cross_product=use_cross_product,
                                  transform_to_earth=transform_to_earth)

    ####################################################################################################################
    # Calculate time shift between radar and ship and shift radar time or seapath time depending on version
    ####################################################################################################################
    chirp_ts = calc_chirp_timestamps(data['VHSpec']['ts'], date, version='center')
    rg_borders = get_range_bin_borders(3, data)
    # transform bin boundaries, necessary because python starts counting at 0
    rg_borders_id = rg_borders - np.array([0, 1, 1, 1])
    # setting the length of the mean doppler velocity time series for calculating time shift
    n_ts_run = np.int(10 * 60 / 1.5)  # 10 minutes with time res of 1.5 s
    if version == 'ca':
        # here seapath is a xarray DataSet
        seapath = seapath.dropna('time_shifted')  # drop nans for interpolation
        seapath_time = seapath['time_shifted'].values.astype(float) / 10 ** 9  # get nan free time in seconds
    elif version == 'jr':
        # here seapath is a pandas DataFrame
        seapath = seapath.dropna()
        seapath_time = seapath.index.values.astype(float) / 10 ** 9  # get nan free time in seconds
    # prepare interpolation function for angular velocity
    Cs = CubicSpline(seapath_time, seapath['heave_rate_radar'])
    plot_path = f'/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/cloudnet_input_heave_cor_{version}/time_shift_plots'
    delta_t_min = -3.  # minimum time shift
    delta_t_max = 3.  # maximum time shift

    # find a 10 minute mdv time series in every hour of radar data and for each chirp if possible
    # calculate time shift for each hour and each chirp
    chirp_ts_shifted, time_shift_array = calc_shifted_chirp_timestamps(data['mdv']['ts'], data['mdv']['var'],
                                                                       chirp_ts, rg_borders_id, n_ts_run, Cs,
                                                                       no_chirps=3, pathFig=plot_path,
                                                                       delta_t_min=delta_t_min,
                                                                       delta_t_max=delta_t_max,
                                                                       date=date, plot_fig=True)

    if shift != 0:
        seapath = shift_seapath(seapath, shift)
    else:
        logger.debug(f"Shift is {shift}! Seapath data is not shifted!")
    ####################################################################################################################
    # Calculating heave correction array and translate to number of Doppler bin shifts
    ####################################################################################################################
    if version == 'ca':
        # calculate the correction matrix
        heave_corr = calc_corr_matrix_claudia(data['mdv']['ts'], data['mdv']['rg'], rg_borders_id, chirp_ts_shifted, Cs)
        seapath_out = seapath
    elif version == 'jr':
        # make input container for calc_heave_corr function
        container = {'C1Range': data['C1Range'], 'C2Range': data['C2Range'], 'C3Range': data['C3Range'],
                     'SeqIntTime': data['SeqIntTime'], 'ts': data['VHSpec']['ts'], 'MaxVel': data['MaxVel'],
                     'DoppLen': data["DoppLen"]}
        heave_corr, seapath_out = calc_heave_corr(container, chirp_ts_shifted, seapath, mean_hr=mean_hr)

    no_chirps = len(data['DoppLen'])
    range_bins = get_range_bin_borders(no_chirps, data)
    doppler_res = calc_dopp_res(data['MaxVel'], data['DoppLen'], no_chirps, range_bins)

    n_dopp_bins_shift, heave_corr = heave_rate_to_spectra_bins(heave_corr, doppler_res)

    ####################################################################################################################
    # Shifting spectra and writing to new 3D array
    ####################################################################################################################

    try:
        # correct spectra for heave rate by moving it by the corresponding number of Doppler bins
        spectra = data['VHSpec']['var']
        new_spectra = np.empty_like(spectra)
        for iT in range(data['n_ts']):
            # loop through time steps
            for iR in range(data['n_rg']):
                # loop through range gates
                # TODO: check if mask is True and skip, although masked shifted spectra do not introduce any error,
                # this might speed up things...
                try:
                    shift = int(n_dopp_bins_shift[iT, iR])
                except ValueError:
                    logger.debug(f"shift at [{iT}, {iR}] is NaN, set to zero")
                    shift = 0
                spectrum = spectra[iT, iR, :]
                if add:
                    new_spec = np.roll(spectrum, shift)
                elif not add:
                    new_spec = np.roll(spectrum, -shift)

                new_spectra[iT, iR, :] = new_spec

        logger.info(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_spectra, heave_corr, n_dopp_bins_shift, seapath_out, time_shift_array
    except KeyError:
        logger.info(f"No input spectra found! Cannot shift spectra.\n Returning only heave_corr and n_dopp_bins_shift array!")
        logger.info(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
        new_spectra = ["I'm an empty list!"]  # create an empty list to return the same number of variables
        return new_spectra, heave_corr, n_dopp_bins_shift, seapath_out, time_shift_array


def read_seapath(date, path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                 **kwargs):
    """
    Read in daily Seapath measurements from RV Meteor from .dat files to a pandas.DataFrame
    Args:
        date (datetime.datetime): object with date of current file
        path (str): path to seapath files
        kwargs for read_csv
            output_format (str): whether a pandas data frame or a xarray dataset is returned

    Returns:
        seapath (DataFrame): DataFrame with Seapath measurements

    """
    # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
    start = time.time()
    # unpack kwargs
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    output_format = kwargs['output_format'] if 'output_format' in kwargs else 'pandas'
    if date < datetime.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=skiprows, na_values=-999.00,
                          index_col='date time', nrows=nrows)
    # transform index to datetime
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'time'
    seapath.columns = ['yaw', 'heave', 'pitch', 'roll']  # rename columns
    logger.info(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")
    if output_format == 'pandas':
        pass
    elif output_format == 'xarray':
        seapath = seapath.to_xarray()

    return seapath


def read_dship(date, **kwargs):
    """Read in 1 Hz DSHIP data and return pandas DataFrame

    Args:
        date (str): yyyymmdd (eg. 20200210)
        **kwargs: kwargs for pd.read_csv (not all implemented) https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns: 
        pd.DataFrame with 1 Hz DSHIP data

    """
    tstart = time.time()
    path = kwargs['path'] if 'path' in kwargs else "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    cols = kwargs['cols'] if 'cols' in kwargs else None  # always keep the 0th column (datetime column)
    file = f"{path}/RV-Meteor_DSHIP_all_1Hz_{date}.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement, set index column
    df = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=skiprows, index_col='date time', nrows=nrows,
                     usecols=cols)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    logger.info(f"Done reading in DSHIP data in {time.time() - tstart:.2f} seconds")

    return df


def f_shiftTimeDataset(dataset):
    """
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : function to shift time variable of the dataset to the central value of the time interval
    of the time step
    input:
        dataset: xarray dataset
    output:
        dataset: xarray dataset with the time coordinate shifted added to the coordinates and the variables now referring to the shifted time array
    """
    # reading time array
    time = dataset['time'].values
    # calculating deltaT using consecutive time stamps
    deltaT = time[2] - time[1]
    # print('delta T for the selected dataset: ', deltaT)
    # defining additional coordinate to the dataset
    dataset.coords['time_shifted'] = dataset['time'] + 0.5 * deltaT
    # exchanging coordinates in the dataset
    datasetNew = dataset.swap_dims({'time': 'time_shifted'})
    return (datasetNew)


def calc_heave_rate(seapath, x_radar=-11, y_radar=4.07, z_radar=15.8, only_heave=False, use_cross_product=True,
                    transform_to_earth=True):
    """
    Calculate heave rate at a certain location of a ship with the measurements of the INS

    Args:
        seapath (pd.DataFrame): Data frame with heading, roll, pitch and heave as columns
        x_radar (float): x position of location with respect to INS in meters
        y_radar (float): y position of location with respect to INS in meters
        z_radar (float): z position of location with respect to INS in meters
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as described in https://repository.library.noaa.gov/view/noaa/17400

    Returns:
        seapath (pd.DataFrame): Data frame as input with additional columns radar_heave, pitch_heave, roll_heave and
        "heave_rate"

    """
    t1 = time.time()
    logger.info("Calculating Heave Rate...")
    # angles in radians
    pitch = np.deg2rad(seapath["pitch"])
    roll = np.deg2rad(seapath["roll"])
    yaw = np.deg2rad(seapath["yaw"])
    # time delta between two time steps in seconds
    d_t = np.ediff1d(seapath.index).astype('float64') / 1e9
    if not use_cross_product:
        logger.info("using a simple geometric approach")
        if not only_heave:
            logger.info("using also the roll and pitch induced heave")
            pitch_heave = x_radar * np.tan(pitch)
            roll_heave = y_radar * np.tan(roll)

        elif only_heave:
            logger.info("using only the ships heave")
            pitch_heave = 0
            roll_heave = 0

        # sum up heave, pitch induced and roll induced heave
        seapath["radar_heave"] = seapath["heave"] + pitch_heave + roll_heave
        # add pitch and roll induced heave to data frame to include in output for quality checking
        seapath["pitch_heave"] = pitch_heave
        seapath["roll_heave"] = roll_heave
        # ediff1d calculates the difference between consecutive elements of an array
        # heave difference / time difference = heave rate
        heave_rate = np.ediff1d(seapath["radar_heave"]) / d_t

    else:
        logger.info("Using the cross product approach from Hannes Griesche")
        # change of angles with time
        d_roll = np.ediff1d(roll) / d_t  # phi
        d_pitch = np.ediff1d(pitch) / d_t  # theta
        d_yaw = np.ediff1d(yaw) / d_t  # psi
        seapath_heave_rate = np.ediff1d(seapath["heave"]) / d_t  # heave rate at seapath
        pos_radar = np.array([x_radar, y_radar, z_radar])  # position of radar as a vector
        ang_rate = np.array([d_roll, d_pitch, d_yaw]).T  # angle velocity as a matrix
        pos_radar_exp = np.tile(pos_radar, (ang_rate.shape[0], 1))  # expand to shape of ang_rate
        cross_prod = np.cross(ang_rate, pos_radar_exp)  # calculate cross product

        if transform_to_earth:
            logger.info("Transform into Earth Coordinate System")
            phi, theta, psi = roll, pitch, yaw
            a1 = np.cos(theta) * np.cos(psi)
            a2 = -1 * np.cos(phi) * np.sin(psi) + np.sin(theta) * np.cos(psi) * np.sin(phi)
            a3 = np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)
            b1 = np.cos(theta) * np.sin(psi)
            b2 = np.cos(phi) * np.cos(psi) + np.sin(theta) * np.sin(phi) * np.sin(psi)
            b3 = -1 * np.cos(psi) * np.sin(phi) + np.cos(phi) * np.sin(theta) * np.sin(psi)
            c1 = -1 * np.sin(theta)
            c2 = np.cos(theta) * np.sin(phi)
            c3 = np.cos(theta) * np.cos(phi)
            Q_T = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
            # remove first entry of Q_T to match dimension of cross_prod
            Q_T = Q_T[:, :, 1:]
            cross_prod = np.einsum('ijk,kj->kj', Q_T, cross_prod)

        heave_rate = seapath_heave_rate + cross_prod[:, 2]  # calculate heave rate

    # add heave rate to seapath data frame
    # the first calculated heave rate corresponds to the second time step
    heave_rate = pd.DataFrame({'heave_rate_radar': heave_rate}, index=seapath.index[1:])
    seapath = seapath.join(heave_rate)

    logger.info(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")
    return seapath

def f_calcRMatrix(rollShipArr, pitchShipArr, yawShipArr, NtimeShip):
    """
    author: Claudia Acquistapace
    date : 27/10/2020
    goal: function to calculate R matrix given roll, pitch, yaw
    input:
        rollShipArr: roll array in degrees
        pitchShipArr: pitch array in degrees
        yawShipArr: yaw array in degrees
        NtimeShip: dimension of time array for the definition of R_inv as [3,3,dimTime]
    output:
        R[3,3,Dimtime]: array of rotational matrices, one for each time stamp
    """
    # calculation of the rotational matrix for each time stamp of the ship data for the day
    cosTheta = np.cos(np.deg2rad(rollShipArr))
    senTheta = np.sin(np.deg2rad(rollShipArr))
    cosPhi = np.cos(np.deg2rad(pitchShipArr))
    senPhi = np.sin(np.deg2rad(pitchShipArr))
    cosPsi = np.cos(np.deg2rad(yawShipArr))
    senPsi = np.sin(np.deg2rad(yawShipArr))

    R = np.zeros([3, 3, NtimeShip])
    A = np.zeros([3, 3, NtimeShip])
    B = np.zeros([3, 3, NtimeShip])
    C = np.zeros([3, 3, NtimeShip])
    R.fill(np.nan)
    A.fill(0.)
    B.fill(0.)
    C.fill(0.)

    # indexing for the matrices
    # [0,0]  [0,1]  [0,2]
    # [1,0]  [1,1]  [1,2]
    # [2,0]  [2,1]  [2,2]
    A[0, 0, :] = 1
    A[1, 1, :] = cosTheta
    A[1, 2, :] = -senTheta
    A[2, 1, :] = senTheta
    A[2, 2, :] = cosTheta

    B[0, 0, :] = cosPhi
    B[1, 1, :] = 1
    B[2, 2, :] = cosPhi
    B[0, 2, :] = senPhi
    B[2, 0, :] = -senPhi

    C[0, 0, :] = cosPsi
    C[0, 1, :] = -senPsi
    C[2, 2, :] = 1
    C[1, 0, :] = senPsi
    C[1, 1, :] = cosPsi

    # calculation of the rotation matrix
    A = np.moveaxis(A, 2, 0)
    B = np.moveaxis(B, 2, 0)
    C = np.moveaxis(C, 2, 0)
    R = np.matmul(C, np.matmul(B, A))
    R = np.moveaxis(R, 0, 2)
    return R


def calc_heave_rate_claudia(data, x_radar=-11, y_radar=4.07, z_radar=-15.8):
    """Calculate heave rate at a certain location on a ship according to Claudia Acquistapace's approach

    Args:
        data (xr.DataSet): Data Set with heading, roll, pitch and heave as columns
        x_radar (float): x position of location with respect to INS in meters
        y_radar (float): y position of location with respect to INS in meters
        z_radar (float): z position of location with respect to INS in meters

    Returns: xr.DataSet with additional variable heave_rate

    """
    r_radar = [x_radar, y_radar, z_radar]
    # calculation of w_ship
    heave = data['heave'].values
    timeShip = data['time_shifted'].values.astype('float64') / 10 ** 9
    w_ship = np.diff(heave, prepend=np.nan) / np.diff(timeShip, prepend=np.nan)

    # calculating rotational terms
    roll = data['roll'].values
    pitch = data['pitch'].values
    yaw = data['yaw'].values
    NtimeShip = len(timeShip)
    r_ship = np.zeros((3, NtimeShip))

    # calculate the position of the  radar on the ship r_ship:
    R = f_calcRMatrix(roll, pitch, yaw, NtimeShip)
    for i in range(NtimeShip):
        r_ship[:, i] = np.dot(R[:, :, i], r_radar)

    # calculating vertical component of the velocity of the radar on the ship (v_rot)
    w_rot = np.diff(r_ship[2, :], prepend=np.nan) / np.diff(timeShip, prepend=np.nan)

    # calculating total ship velocity at radar
    heave_rate = w_rot + w_ship
    data['w_rot'] = (('time_shifted'), w_rot)
    data['heave_rate'] = (('time_shifted'), w_ship)
    data['heave_rate_radar'] = (('time_shifted',), heave_rate)

    return data


def find_mdv_time_series(mdv_values, radar_time, n_ts_run):
    """
    author: Claudia Acquistapace, Johannes Roettenbacher
    Identify, given a mean doppler velocity matrix, a sequence of length n_ts_run of values in the matrix
    at a given height that contains the minimum possible amount of nan values in it.

    Args:
        mdv_values (ndarray): time x heigth matrix of Doppler Velocity
        radar_time (ndarray): corresponding radar time stamps in seconds (unix time)
        n_ts_run (int): number of timestamps needed in a mdv series

    Returns:
        valuesTimeSerie (ndarray): time series of Doppler velocity with length n_ts_run
        time_series (ndarray): corresponding time stamps to Doppler velocity time series
        i_height_sel (int): index of chosen height
        valuesColumnMean (ndarray): time series of mean Doppler velocity averaged over height with length n_ts_run

    """
    #  concept: scan the matrix using running mean for every height, and check the number of nans in the selected serie.
    nanAmountMatrix = np.zeros((mdv_values.shape[0] - n_ts_run, mdv_values.shape[1]))
    nanAmountMatrix.fill(np.nan)
    for indtime in range(mdv_values.shape[0] - n_ts_run):
        mdvChunk = mdv_values[indtime:indtime + n_ts_run, :]
        # count number of nans in each height
        nanAmountMatrix[indtime, :] = np.sum(np.isnan(mdvChunk), axis=0)

    # find indeces where nanAmount is minimal
    ntuples = np.where(nanAmountMatrix == np.nanmin(nanAmountMatrix))
    i_time_sel = ntuples[0][0]
    i_height_sel = ntuples[1][0]

    # extract corresponding time series of mean Doppler velocity values for the chirp
    valuesTimeSerie = mdv_values[i_time_sel:i_time_sel + n_ts_run, i_height_sel]
    time_series = radar_time[i_time_sel:i_time_sel + n_ts_run]

    ###### adding test for columns ########
    valuesColumn = mdv_values[i_time_sel:i_time_sel + n_ts_run, :]
    valuesColumnMean = np.nanmean(valuesColumn, axis=1)

    return valuesTimeSerie, time_series, i_height_sel, valuesColumnMean


def calc_time_shift(w_radar_meanCol, delta_t_min, delta_t_max, resolution, w_ship_chirp, timeSerieRadar, pathFig, chirp,
                    hour, date):
    """
    author: Claudia Acquistapace, Jan. H. Schween, Johannes Roettenbacher
    goal:   calculate and estimation of the time lag between the radar time stamps and the ship time stamp

    NOTE: adding or subtracting the obtained time shift depends on what you did
    during the calculation of the covariances: if you added/subtracted time _shift
    to t_radar you have to do the same for the 'exact time'
    Here is the time shift analysis as plot:
    <ww> is short for <w'_ship*w'_radar> i.e. covariance between vertical speeds from
    ship movements and radar its maximum gives an estimate for optimal agreement in
    vertical velocities of ship and radar
    <Delta w^2> is short for <(w[i]-w[i-1])^2> where w = w_rad - 2*w_ship - this
    is a measure for the stripeness. Its minimum gives an
    estimate how to get the smoothest w data
    Args:
        w_radar_meanCol (ndarray): time series of mean Doppler velocity averaged over height with no nan values
        delta_t_min (float): minimum time shift
        delta_t_max (float): maximum time shift
        resolution (float): time step by which to increment possible time shift
        w_ship_chirp (ndarray): vertical velocity of the radar at the exact chirp time step
        timeSerieRadar (ndarray): time stamps of the mean Doppler velocity time series (w_radar_meanCol)
        pathFig (str): file path where figures should be stored
        chirp (int): which chirp is being processed
        hour (int): which hour of the day is being processed (0-23)
        date (datetime): which day is being processed

    Returns: time shift between radar data and ship data in seconds, quicklooks for each calculation
    """
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    plt.gcf().subplots_adjust(bottom=0.15)

    # calculating variation for w_radar
    w_prime_radar = w_radar_meanCol - np.nanmean(w_radar_meanCol)

    # calculating covariance between w-ship and w_radar where w_ship is shifted for each deltaT given by DeltaTimeShift
    DeltaTimeShift = np.arange(delta_t_min, delta_t_max, step=resolution)
    cov_ww = np.zeros(len(DeltaTimeShift))
    deltaW_ship = np.zeros(len(DeltaTimeShift))

    for i in range(len(DeltaTimeShift)):
        # calculate w_ship interpolating it on the new time array (timeShip+deltatimeShift(i))
        T_corr = timeSerieRadar + DeltaTimeShift[i]

        # interpolating w_ship on the shifted time series
        cs_ship = CubicSpline(timeSerieRadar, w_ship_chirp)
        w_ship_shifted = cs_ship(T_corr)

        # calculating w_prime_ship with the new interpolated series
        w_ship_prime = w_ship_shifted - np.nanmean(w_ship_shifted)

        # calculating covariance of the prime series
        cov_ww[i] = np.nanmean(w_ship_prime * w_prime_radar)

        # calculating sharpness deltaW_ship
        w_corrected = w_radar_meanCol - w_ship_shifted
        delta_w = (np.ediff1d(w_corrected)) ** 2
        deltaW_ship[i] = np.nanmean(delta_w)

    # calculating max of covariance and min of deltaW_ship
    minDeltaW = np.nanmin(deltaW_ship)
    indMin = np.where(deltaW_ship == minDeltaW)[0][0]
    maxCov_w = np.nanmax(cov_ww)
    indMax = np.where(cov_ww == maxCov_w)[0][0]
    try:
        logger.info(f'Time shift found for chirp {chirp} at hour {hour}: {DeltaTimeShift[indMin]}')
        # calculating time shift for radar data
        timeShift_chirp = DeltaTimeShift[indMin]
        # if time shift is equal delta_t_min it's probably false -> set it to 0
        if np.abs(timeShift_chirp) == np.abs(delta_t_min):
            timeShift_chirp = 0

        # plot results
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.plot(DeltaTimeShift, cov_ww, color='red', linestyle=':', label='cov_ww')
        ax.axvline(x=DeltaTimeShift[indMax], color='red', linestyle=':', label='max cov_w')
        ax.plot(DeltaTimeShift, deltaW_ship, color='red', label='Deltaw^2')
        ax.axvline(x=DeltaTimeShift[indMin], color='red', label='min Deltaw^2')
        ax.legend(frameon=False)
        # ax.xaxis_date()
        ax.set_ylim(-0.1, 2.)  # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
        ax.set_xlim(delta_t_min, delta_t_max)  # limits of the x-axes
        ax.set_title(
            f'Covariance and Sharpiness for chirp {chirp}: {date:%Y-%m-%d} hour: {hour}, '
            f'time lag found : {DeltaTimeShift[indMin]}',
            fontsize=fontSizeTitle, loc='left')
        ax.set_xlabel("Time Shift [seconds]", fontsize=fontSizeX)
        ax.set_ylabel('w [m s$^{-1}$]', fontsize=fontSizeY)
        fig.tight_layout()
        fig.savefig(f'{pathFig}/{date:%Y%m%d}_timeShiftQuicklook_chirp{chirp}_hour{hour}.png', format='png')
        plt.close()
    except IndexError:
        logger.info(f'Not enough data points for time shift calculation in chirp {chirp} at hour {hour}!')
        timeShift_chirp = 0

    return timeShift_chirp


def calc_chirp_timestamps(radar_ts, date, version):
    """ Calculate the exact timestamp for each chirp corresponding with the center or start of the chirp
    The timestamp in the radar file corresponds to the end of a chirp sequence with an accuracy of 0.1 s

    Args:
        radar_ts (ndarray): timestamps of the radar with milliseconds in seconds
        date (datetime.datetime): date which is being processed
        version (str): should the timestamp correspond to the 'center' or the 'start' of the chirp

    Returns: dict with chirp timestamps

    """
    # make lookup table for chirp durations for each chirptable (see projekt1/remsens/hardware/LIMRAD94/chirptables)
    chirp_durations = pd.DataFrame({"Chirp_No": (1, 2, 3), "tradewindCU": (1.022, 0.947, 0.966),
                                    "Doppler1s": (0.239, 0.342, 0.480), "Cu_small_Tint": (0.225, 0.135, 0.181),
                                    "Cu_small_Tint2": (0.562, 0.572, 0.453)})
    # calculate start time of each chirp by subtracting the duration of the later chirp(s) + the chirp itself
    # the timestamp then corresponds to the start of the chirp
    # select chirp durations according to date
    if date < datetime.datetime(2020, 1, 29, 18, 0, 0):
        chirp_dur = chirp_durations["tradewindCU"]
    elif date < datetime.datetime(2020, 1, 30, 15, 3, 0):
        chirp_dur = chirp_durations["Doppler1s"]
    elif date < datetime.datetime(2020, 1, 31, 22, 28, 0):
        chirp_dur = chirp_durations["Cu_small_Tint"]
    else:
        chirp_dur = chirp_durations["Cu_small_Tint2"]

    chirp_timestamps = dict()
    if version == 'center':
        chirp_timestamps["chirp_1"] = radar_ts - chirp_dur[0] - chirp_dur[1] - chirp_dur[2] / 2
        chirp_timestamps["chirp_2"] = radar_ts - chirp_dur[1] - chirp_dur[2] / 2
        chirp_timestamps["chirp_3"] = radar_ts - chirp_dur[2] / 2
    else:
        chirp_timestamps["chirp_1"] = radar_ts - chirp_dur[0] - chirp_dur[1] - chirp_dur[2]
        chirp_timestamps["chirp_2"] = radar_ts - chirp_dur[1] - chirp_dur[2]
        chirp_timestamps["chirp_3"] = radar_ts - chirp_dur[2]

    return chirp_timestamps


def calc_shifted_chirp_timestamps(radar_ts, radar_mdv, chirp_ts, rg_borders_id, n_ts_run, Cs_w_radar, **kwargs):
    """
    Calculates the time shift between each chirp time stamp and the ship time stamp for every hour and every chirp.
    Args:
        radar_ts (ndarray): radar time stamps in seconds (unix time)
        radar_mdv (ndarray): time x height matrix of mean Doppler velocity from radar
        chirp_ts (ndarray): exact chirp time stamps
        rg_borders_id (ndarray): indices of chirp boundaries
        n_ts_run (int): number of time steps necessary for mean Doppler velocity time series
        Cs_w_radar (scipy.interpolate.CubicSpline): function of vertical velocity of radar against time
        **kwargs:
            no_chirps (int): number of chirps in radar measurement
            plot_fig (bool): plot quicklook

    Returns: time shifted chirp time stamps, array with time shifts for each chirp and hour

    """
    no_chirps = kwargs['no_chirps'] if 'no_chirps' in kwargs else 3
    delta_t_min = kwargs['delta_t_min'] if 'delta_t_min' in kwargs else radar_ts[0] - radar_ts[1]
    delta_t_max = kwargs['delta_t_max'] if 'delta_t_max' in kwargs else radar_ts[1] - radar_ts[0]
    resolution = kwargs['resolution'] if 'resolution' in kwargs else 0.05
    pathFig = kwargs['pathFig'] if 'pathFig' in kwargs else "./tmp"
    date = kwargs['date'] if 'date' in kwargs else pd.to_datetime(radar_ts[0], unit='s')
    plot_fig = kwargs['plot_fig'] if 'plot_fig' in kwargs else False

    time_shift_array = np.zeros((len(radar_ts), no_chirps))
    chirp_ts_shifted = chirp_ts
    # get total hours in data and then loop through each hour
    hours = np.int(np.ceil(radar_ts.shape[0] * np.mean(np.diff(radar_ts)) / 60 / 60))
    idx = np.int(np.floor(len(radar_ts) / hours))
    for i in range(hours):
        start_idx = i * idx
        if i < hours-1:
            end_idx = (i + 1) * idx
        else:
            end_idx = time_shift_array.shape[0]
        for j in range(no_chirps):
            # set time and range slice
            ts_slice, rg_slice = slice(start_idx, end_idx), slice(rg_borders_id[j], rg_borders_id[j + 1])
            mdv_slice = radar_mdv[ts_slice, rg_slice]
            time_slice = chirp_ts[f'chirp_{j + 1}'][
                ts_slice]  # select the corresponding exact chirp time for the mdv slice
            mdv_series, time_mdv_series, height_id, mdv_mean_col = find_mdv_time_series(mdv_slice, time_slice,
                                                                                        n_ts_run)

            # selecting w_radar values of the chirp over the same time interval as the mdv_series
            w_radar_chirpSel = Cs_w_radar(time_mdv_series)

            # calculating time shift for the chirp and hour if at least n_ts_run measurements are available
            if np.sum(~np.isnan(mdv_mean_col)) == n_ts_run:
                time_shift_array[ts_slice, j] = calc_time_shift(mdv_mean_col, delta_t_min, delta_t_max, resolution,
                                                                w_radar_chirpSel, time_mdv_series,
                                                                pathFig, j + 1, i, date)

            # recalculate exact chirp time including time shift due to lag
            chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice] = chirp_ts[f'chirp_{j + 1}'][ts_slice] - time_shift_array[
                ts_slice, j]
            # get w_radar at the time shifted exact chirp time stamps
            w_radar_exact = Cs_w_radar(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice])

            if plot_fig:
                # plot mdv time series and shifted radar heave rate
                ts_idx = [argnearest(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice], t) for t in time_mdv_series]
                plot_time = pd.to_datetime(time_mdv_series, unit='s')
                plot_df = pd.DataFrame(dict(time=plot_time, mdv_mean_col=mdv_mean_col,
                                            w_radar_org=Cs_w_radar(time_mdv_series),
                                            w_radar_chirpSel=w_radar_chirpSel,
                                            w_radar_exact_shifted=w_radar_exact[ts_idx])).set_index('time')
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.plot(plot_df['mdv_mean_col'], color='red', label='mean mdv over column at original radar time')
                ax.plot(plot_df['w_radar_org'], color='blue', linewidth=0.2, label='w_radar at original radar time')
                ax.plot(plot_df['w_radar_chirpSel'], color='blue', label='w_radar at original chirp time')
                ax.plot(plot_df['w_radar_exact_shifted'], '.', color='green', label='w_radar shifted')
                ax.set_ylim(-4., 2.)
                ax.legend(frameon=False)
                # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
                ax.set_title(
                    f'Velocity for Time Delay Calculations : {date:%Y-%m-%d} shift = {time_shift_array[start_idx, j]}',
                    loc='left')
                ax.set_xlabel("Time [day hh:mm]")
                ax.set_ylabel('w [m s$^{-1}$]')
                ax.xaxis_date()
                ax.grid()
                fig.autofmt_xdate()
                fig.savefig(f'{pathFig}/{date:%Y%m%d}_time-series_mdv_w-radar_chirp{j + 1}_hour{i}.png')
                plt.close()

    return chirp_ts_shifted, time_shift_array


def calc_corr_matrix_claudia(radar_ts, radar_rg, rg_borders_id, chirp_ts_shifted, Cs_w_radar):
    """
    Calculate the correction matrix to correct the mean Doppler velocity for the ship vertical motion.
    Args:
        radar_ts (ndarray): original radar time stamps in seconds (unix time)
        radar_rg (ndarray): radar range gates
        rg_borders_id (ndarray): indices of chirp boundaries
        chirp_ts_shifted (dict): hourly shifted chirp time stamps
        Cs_w_radar (scipy.interpolate.CubicSpline): function of vertical velocity of radar against time

    Returns: correction matrix for mean Doppler velocity

    """
    no_chirps = len(chirp_ts_shifted)
    corr_matrix = np.zeros((len(radar_ts), len(radar_rg)))
    # get total hours in data and then loop through each hour
    hours = np.int(np.ceil(radar_ts.shape[0] * np.mean(np.diff(radar_ts)) / 60 / 60))
    # divide the day in equal hourly slices
    idx = np.int(np.floor(len(radar_ts) / hours))
    for i in range(hours):
        start_idx = i * idx
        if i < hours-1:
            end_idx = (i + 1) * idx
        else:
            end_idx = len(radar_ts)
        for j in range(no_chirps):
            # set time and range slice
            ts_slice, rg_slice = slice(start_idx, end_idx), slice(rg_borders_id[j], rg_borders_id[j + 1])
            # get w_radar at the time shifted exact chirp time stamps
            w_radar_exact = Cs_w_radar(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice])
            # add a dimension to w_radar_exact and repeat it over this dimension (range) to fill the hour and
            # chirp of the correction array
            tmp = np.repeat(np.expand_dims(w_radar_exact, 1), rg_borders_id[j + 1] - rg_borders_id[j], axis=1)
            corr_matrix[ts_slice, rg_slice] = tmp

    return corr_matrix


def get_range_bin_borders(no_chirps, container):
    """get the range bins which correspond to the chirp borders of a FMCW radar

    Args:
        no_chirps (int): Number of chirps
        container (dict): Dictionary with C1/2/3Range variable from LV1 files

    Returns: 
        ndarray with chirp borders including 0 range_bins

    """
    range_bins = np.zeros(no_chirps + 1, dtype=np.int)  # needs to be length 4 to include all +1 chirp borders
    for i in range(no_chirps):
        try:
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'][0].shape
        except ValueError:
            # in case only one file is read in data["C1Range"]["var"] has only one dimension
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'].shape

    return range_bins


def calc_heave_corr(container, chirp_ts, seapath, mean_hr=True):
    """Calculate heave correction for mean Doppler velocity

    Args:
        container (larda container): LIMRAD94 C1/2/3_Range, SeqIntTime, ts
        chirp_ts (dict): dictionary with exact radar chirp time stamps
        seapath (pd.DataFrame): Data frame with heave rate column ("heave_rate_radar")
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp

    Returns: 
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, time x range

    """
    start = time.time()
    ####################################################################################################################
    # Calculating Timestamps for each chirp
    ####################################################################################################################
    # array with range bin numbers of chirp borders
    no_chirps = len(chirp_ts)
    range_bins = get_range_bin_borders(no_chirps, container)

    seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    total_range_bins = range_bins[-1]  # get total number of range bins
    # initialize output variables
    heave_corr = np.empty(shape=(container["ts"].shape[0], total_range_bins))  # time x range
    seapath_out = pd.DataFrame()
    for i in range(no_chirps):
        t1 = time.time()
        # get integration time for chirp
        int_time = pd.Timedelta(seconds=container['SeqIntTime'][i])
        # convert timestamps of moments to array
        ts = chirp_ts[f"chirp_{i+1}"].data
        id_diff_mins = []  # initialize list for indices of the time steps with minimum difference
        means_ls = []  # initialize list for means over integration time for each radar time step
        for t in ts:
            id_diff_min = argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
            id_diff_mins.append(id_diff_min)
            # get time stamp of closest index
            ts_id_diff_min = seapath.index[id_diff_min]
            if mean_hr:
                # select rows from closest time stamp to end of integration time and average, append to list
                means_ls.append(seapath[ts_id_diff_min:ts_id_diff_min+int_time].mean())
            else:
                means_ls.append(seapath.loc[ts_id_diff_min])

        # concatenate all means into one dataframe with the original header (transpose)
        seapath_closest = pd.concat(means_ls, axis=1).T
        # add index with closest seapath time step to radar time step
        seapath_closest.index = seapath.index[id_diff_mins]

        # check if heave rate is greater than 5 standard deviations away from the daily mean and filter those values
        # by averaging the step before and after
        std = np.nanstd(seapath_closest["heave_rate_radar"])
        # try to get indices from values which do not pass the filter. If that doesn't work, then there are no values
        # which don't pass the filter and a ValueError is raised. Write this to a logger
        try:
            id_max = np.asarray(np.abs(seapath_closest["heave_rate_radar"]) > 5 * std).nonzero()[0]
            for j in range(len(id_max)):
                idc = id_max[j]
                warnings.warn(f"Heave rate greater 5 * std encountered ({seapath_closest['heave_rate_radar'][idc]})! \n"
                              f"Using average of step before and after. Index: {idc}", UserWarning)
                avg_hrate = (seapath_closest["heave_rate_radar"][idc - 1] + seapath_closest["heave_rate_radar"][idc + 1]) / 2
                if avg_hrate > 5 * std:
                    warnings.warn(f"Heave Rate value greater than 5 * std encountered ({avg_hrate})! \n"
                                  f"Even after averaging step before and after too high value! Index: {idc}",
                                  UserWarning)
                seapath_closest["heave_rate_radar"][idc] = avg_hrate
        except ValueError:
            logging.info(f"All heave rate values are within 5 standard deviation of the daily mean!")

        # add column with chirp number to distinguish in quality control
        seapath_closest["Chirp_no"] = np.repeat(i + 1, len(seapath_closest.index))
        # make data frame with used heave rates
        seapath_out = seapath_out.append(seapath_closest)
        # create array with same dimensions as velocity (time, range)
        heave_rate = np.expand_dims(seapath_closest["heave_rate_radar"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins of the chirp
        shape = range_bins[i + 1] - range_bins[i]
        heave_corr[:, range_bins[i]:range_bins[i+1]] = heave_rate.repeat(shape, axis=1)
        logger.info(f"Calculated heave correction for Chirp {i+1} in {time.time() - t1:.2f} seconds")

    logger.info(f"Done with heave correction calculation in {time.time() - start:.2f} seconds")
    return heave_corr, seapath_out


def calc_dopp_res(MaxVel, DoppLen, no_chirps, range_bins):
    """

    Args:
        MaxVel (ndarray): Unambiguous Doppler velocity (+/-) m/s from LV1 file
        DoppLen (ndarray): Number of spectral lines in Doppler spectra from LV1 file
        no_chirps (int): Number of chirps
        range_bins (ndarray): range bin number of lower chirp borders, starts with 0

    Returns: 
        1D array with Doppler resolution for each height bin

    """
    DoppRes = np.divide(2.0 * MaxVel, DoppLen)
    dopp_res = np.empty(range_bins[-1])
    for ic in range(no_chirps):
        dopp_res[range_bins[ic]:range_bins[ic + 1]] = DoppRes[ic]
    return dopp_res


def heave_rate_to_spectra_bins(heave_corr, doppler_res):
    """translate the heave correction to Doppler spectra bins

    Args:
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, time x range
        doppler_res (ndarray): Doppler resolution of each chirp of LIMRAD94 for whole range 1 x range

    Returns: 
        ndarray with number of bins to move each Doppler spectrum
        
        - **n_dopp_bins_shift** (*ndarray*): of same dimension as heave_corr
        - **heave_corr**

    """
    start = time.time()
    # add a dimension to the doppler_res vector
    doppler_res = np.expand_dims(doppler_res, axis=1)
    # repeat doppler_res to same time dimension as heave_corr
    doppler_res = np.repeat(doppler_res.T, heave_corr.shape[0], axis=0)

    assert doppler_res.shape == heave_corr.shape, f"Arrays have different shape! {doppler_res.shape} " \
                                                  f"and {heave_corr.shape}"

    # calculate number of Doppler bins
    n_dopp_bins_shift = np.round(heave_corr / doppler_res)
    logger.info(f"Done with translation of heave corrections to Doppler bins in {time.time() - start:.2f} seconds")
    return n_dopp_bins_shift, heave_corr


def shift_seapath(seapath, shift):
    """Shift seapath values by given shift

    Args:
        seapath (pd.Dataframe): Dataframe with heave motion of RV-Meteor
        shift (int): number of time steps to shift data

    Returns: 
        shifted Dataframe

    """
    start = time.time()
    logger.info(f"Shifting seapath data by {shift} time steps.")
    # get day of seapath data
    dt = seapath.index[0]
    # shift seapath data by shift
    seapath_shifted = seapath.shift(periods=shift)

    # replace Nans at start with data from the previous day or from following day
    if shift > 0:
        dt_previous = dt - datetime.timedelta(1)  # get date of previous day
        skiprows = np.arange(1, len(seapath) - shift + 2)  # define rows to skip on read in
        # read in one more row for heave rate calculation
        seapath_previous = read_seapath(dt_previous, nrows=shift + 1, skiprows=skiprows)
        seapath_previous = calc_heave_rate(seapath_previous)
        seapath_previous = seapath_previous.iloc[1:, :]  # remove first row (=nan)
        # remove index and replace with index from original data frame
        seapath_previous = seapath_previous.reset_index(drop=True).set_index(seapath_shifted.iloc[0:shift, :].index)
        seapath_shifted.update(seapath_previous)  # overwrite nan values in shifted data frame
    else:
        dt_following = dt + datetime.timedelta(1)  # get date from following day
        seapath_following = read_seapath(dt_following, nrows=np.abs(shift))
        seapath_following = calc_heave_rate(seapath_following)
        # overwrite nan values
        # leaves in one NaN value because the heave rate of the first time step of a day cannot be calculated
        # one nan is better than many (shift) though, so this is alright
        seapath_following = seapath_following.reset_index(drop=True).set_index(seapath_shifted.iloc[shift:, :].index)
        seapath_shifted.update(seapath_following)  # overwrite nan values in shifted data frame

    logger.info(f"Done with shifting seapath data, elapsed time = {seconds_to_fstring(time.time() - start)} [min:sec]")
    return seapath_shifted


def find_closest_timesteps(df, ts):
    """Find closest time steps in a dataframe to a time series

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex
        ts (ndarray): array with time stamps in unix format (seconds since 1-1-1970)

    Returns: 
        pd.DataFrame with only the closest time steps to ts

    """
    tstart = time.time()
    try:
        assert df.index.inferred_type == 'datetime64', "Dataframe Index is not a DatetimeIndex trying to turn into one"
    except AssertionError:
        df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    df_ts = df.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    df_list = []  # initialize lsit to append df rows closest to input time steps to
    for t in ts:
        id_diff_min = argnearest(df_ts, t)  # find index of nearest dship time step to input time step
        ts_id_diff_min = df.index[id_diff_min]  # get time stamp of closest index
        df_list.append(df.loc[ts_id_diff_min])  # append row to list

    # concatenate all rows into one dataframe with the original header (transpose)
    df_closest = pd.concat(df_list, axis=1).T
    logger.info(f"Done finding closest time steps in {time.time() - tstart:.2f} seconds")

    return df_closest


def spectra2sldr(ZSpec, paraminfo, **kwargs):
    """
    This routine calculates the

    Args:
        ZSpec (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Returns:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

    tstart = time.time()  # initialize variables:

    tspec_lin = ZSpec['VHSpec']['var'].copy()  # - ZSpec[iC]['mean']
    hspec_lin = ZSpec['HSpec']['var'].copy()
    tspec_lin = ZSpec['thresh']['var'].copy()  # - ZSpec[iC]['mean']
    vspec_lin = tspec_lin - hspec_lin
    Revhspec_lin = ZSpec['ReVHSpec']['var'].copy()
    Imvhspec_lin = ZSpec['ImVHSpec']['var'].copy()
    vspec_lin[vspec_lin <= 0.0] = np.nan
    hspec_lin[hspec_lin <= 0.0] = np.nan
    Revhspec_lin[Revhspec_lin <= 0.0] = np.nan
    Imvhspec_lin[Imvhspec_lin <= 0.0] = np.nan

    vhspec_complex = Revhspec_lin + Imvhspec_lin * 1j

    for j in range(vspec_lin.shape[0]):

        Zt = tspec_lin[j, :, :]
        Zh = hspec_lin[j, :, :]
        Zre = Revhspec_lin[j, :, :]
        Zim = Imvhspec_lin[j, :, :]


    """
        ZT  = double(ncread(filename,['C' num2str(i) 'VSpec']));    % In the case of STSR mode, this variable contains the combined reflectivity spectrum
        ZH  = double(ncread(filename,['C' num2str(i) 'HSpec']));    % Spectrum in the horizontal channel
        ZRE = double(ncread(filename,['C' num2str(i) 'ReVHSpec'])); % Real part of the covariance spectrum
        ZIM = double(ncread(filename,['C' num2str(i) 'ImVHSpec'])); % Imaginary part of the covariance spectrum
        
        NV  = double(ncread(filename,['C' num2str(i) 'VNoisePow'])); % Integrated noise in the vertical channel
        NH  = double(ncread(filename,['C' num2str(i) 'HNoisePow'])); % Integrated noise in the horizontal channel
        
        SLDR = zeros(size(ZT,2),size(ZT,3)) * NaN;
        
        for j = 1:size(ZT,3)
            
            Zt = ZT(:,:,j);
            Zh = ZH(:,:,j);
            Zre = ZRE(:,:,j);
            Zim = ZIM(:,:,j);
            
            Nv = NV(:,j);
            Nh = NH(:,j);
        
            if SW >= 540
                Zv = 4 * Zt - Zh - 2 * Zre; % Starting from the software version 5.40, the combined spectrum is normalized by 4
            else
                Zv = 2 * Zt - Zh - 2 * Zre; % In the previous versions the combined spectrum was normalized by 2
            end

            clear Zt

            Nfft = size(Zv,1); % Number of spectral lines

            Nv = Nv/Nfft; % Spectral noise power in each spectral bin
            Nh = Nh/Nfft; % Spectral noise power in each spectral bin

            Nv = repmat(Nv,1,Nfft)';
            Nh = repmat(Nh,1,Nfft)';

            % Method based on Galletti et al. 2011

            % According to (10) in the Galletti's paper the depolarization
            % ratio can be calculated from the degree of polarization in the 
            % case of the reflectio symmetry. In the case of vertical
            % observations the assumption is typically reasonable. The
            % assumption is sometimes not applicable in thunderstorm clouds,
            % where electrical activity can allign ice particles in a certain
            % direction. This formula also cannot be used at lower elevation
            % angles since liquid and  ice particles tend to orient
            % horizontally.

            % In the STSR radars the degree of polarization is related to the
            % correlation coefficient (see eq. (12) in Galletti and Zrnic 2012).
            % In the case of vertical observations ZDR is most of the time 1
            % (or 0 dB). In this case the degree of polarization is equal to
            % the correlation coefficient. Thus, in the eq. (10) in the
            % Galletti's paper we can use the correlation coefficient instead
            % of the degree of polarization.

            % The main disadvantage of this method is that it does not take into
            % account that we have signal + noise and we are only interested in
            % the depolarization ratio of the signal. In other words, if SNR is
            % low, there will be strong apparent depolarization caused by noise.
            % In order to avoid this, I have introduced the threshold of 30 dB.
            % The 30 dB threshold has been chosen because this is typically the
            % level of polarimetric coupling in the good antennas of
            % meteorological radars.

            SNRv = Zv./Nv;
            SNRh = Zh./Nh;

            % Spectral lines with less than 30 dB SNR are replaced by NaN
            % values
            k = find((SNRv < 1000) | (SNRh < 1000));

            Zv(k) = NaN;
            Zh(k) = NaN;
            Zre(k) = NaN;
            Zim(k) = NaN;

            Nv(k) = NaN;
            Nh(k) = NaN;

            clear k SNRv SNRh

            Zv = squeeze(nansum(Zv));
            Zh = squeeze(nansum(Zh));
            Zre = squeeze(nansum(Zre));
            Zim = squeeze(nansum(Zim));

            Nv = squeeze(nansum(Nv));
            Nh = squeeze(nansum(Nh));
            
            k = find((Zv ==0) | (Zh == 0));
            
            Zv(k) = NaN;
            Zh(k) = NaN;
            Zre(k) = NaN;
            Zim(k) = NaN;

            rhv  = abs(Zre+1i*Zim)./sqrt((Zv+Nv).*(Zh+Nh)); % Correlation coefficient for each spectral bin
            sldr = ((1-rhv)./(1+rhv));                      % Depolarization ratio according to (10)

            sldr = single(sldr);
            
            SLDR(:,j) = sldr';
            
    """

    logger.info(f'Polarimetric spectra & products calculated, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    return pol



def spectra2polarimetry(ZSpec, paraminfo, **kwargs):
    """
    This routine calculates the

    Args:
        ZSpec (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Returns:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

    tstart = time.time()  # initialize variables:

    vspec_lin = ZSpec['VHSpec']['var'].copy()  # - ZSpec[iC]['mean']
    hspec_lin = ZSpec['HSpec']['var'].copy()
    vspec_lin = vspec_lin - hspec_lin
    Revhspec_lin = ZSpec['ReVHSpec']['var'].copy()
    Imvhspec_lin = ZSpec['ImVHSpec']['var'].copy()
    vspec_lin[vspec_lin <= 0.0] = np.nan
    hspec_lin[hspec_lin <= 0.0] = np.nan
    Revhspec_lin[Revhspec_lin <= 0.0] = np.nan
    Imvhspec_lin[Imvhspec_lin <= 0.0] = np.nan

    vhspec_complex = Revhspec_lin + Imvhspec_lin * 1j

    ZDR = vspec_lin / hspec_lin
    rhoVH = np.absolute(vhspec_complex) / (vspec_lin * hspec_lin)
    phiDP = np.angle(vhspec_complex)
    tmp1 = vspec_lin + hspec_lin - 2 * Revhspec_lin
    tmp2 = vspec_lin + hspec_lin + 2 * Revhspec_lin
    SLDR = tmp1 / tmp2
    CORR = np.absolute(vspec_lin - hspec_lin + 2j * Imvhspec_lin) / np.sqrt(tmp1 * tmp2)

    pol = {'ZDR_s': ZDR, 'ZDR': np.nansum(ZDR, axis=2),
           'rhoVH_s': rhoVH, 'rhoVH': np.nansum(rhoVH, axis=2),
           'phiDP_s': phiDP, 'phiDP': np.nansum(phiDP, axis=2),
           'ldr_s': SLDR, 'ldr': np.nansum(SLDR, axis=2),
           'CORR_s': CORR, 'CORR': np.nansum(CORR, axis=2)
           }

    logger.info(f'Polarimetric spectra & products calculated, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    return pol

