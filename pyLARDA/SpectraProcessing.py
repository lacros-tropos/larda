"""
This routine calculates the radar moments for the RPG 94 GHz FMCW radar 'LIMRAD94' and generates a NetCDF4 file.
The generated files can be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python spec2mom_limrad94.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""
import sys
from numba import jit
import datetime
import copy
import time
import numpy as np
import logging
from datetime import timedelta
import warnings

warnings.simplefilter("ignore", UserWarning)
sys.path.append('../../larda/')

from larda.pyLARDA.helpers import lin2z, z2lin, argnearest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# logger.addHandler(logging.StreamHandler())

@jit(nopython=True, fastmath=True)
def estimate_noise_hs74(spectrum, navg=1, std_div=6.0, nnoise_min=1):
    """REFERENCE TO ARM PYART GITHUB REPO: https://github.com/ARM-DOE/pyart/blob/master/pyart/util/hildebrand_sekhon.py

    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.
    Args:
        spectrum (array): Doppler spectrum in linear units.
        navg (int, optional):  The number of spectral bins over which a moving average has been
            taken. Corresponds to the **p** variable from equation 9 of the
            article. The default value of 1 is appropriate when no moving
            average has been applied to the spectrum.
        std_div (float, optional): Number of standard deviations above mean noise floor to specify the
            signal threshold, default: threshold=mean_noise + 6*std(mean_noise)
        nnoise_min (int, optional): Minimum number of noise samples to consider the estimation valid.

    Returns:
        mean (float): Mean of points in the spectrum identified as noise.
        threshold (float): Threshold separating noise from signal. The point in the spectrum with
            this value or below should be considered as noise, above this value
            signal. It is possible that all points in the spectrum are identified
            as noise. If a peak is required for moment calculation then the point
            with this value should be considered as signal.
        var (float): Variance of the points in the spectrum identified as noise.
        nnoise (int): Number of noise points in the spectrum.
    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.
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
def find_peak_edges(signal, threshold):
    """Returns the indices of left and right edge of the main signal peak in a Doppler spectra.

    Args:
        signal (numpy.array): 1D array Doppler spectra
        threshold: noise threshold

    Returns (list):
        [index_left, index_right]: indices of signal minimum/maximum velocity
    """
    idxMaxSignal = np.argmax(signal)
    len_sig = len(signal)
    index_left, index_right = 0, len_sig

    for ispec in range(idxMaxSignal, len_sig):
        if signal[ispec] < threshold:
            index_right = ispec
            break

    for ispec in range(idxMaxSignal, -1, -1):
        if signal[ispec] < threshold:
            index_left = ispec + 1  # the +1 is important, otherwise a fill_value will corrupt the numba code
            break

    return [index_left, index_right]


def noise_estimation(data, **kwargs):
    """
    Creates a dict containing the noise threshold, mean noise level,
    the variance of the noise, the number of noise values in the spectrum,
    and the boundaries of the main signal peak, if there is one

    Args:
        data (numpy.array): 3D array containing spectra,of dimension (n_ts, n_rg, n_fft)

    Keyword Args:
        **NF (float): threshold = number of standard deviations above mean noise floor, default: 1.0
        **main_peak (float): get the main peaks' index of left and right edge if True, default: True

    Returns:
        dict with noise floor estimation for all time and range points
    """

    n_std = kwargs['NF'] if 'NF' in kwargs else 1.0
    n_avg = kwargs['n_avg'] if 'n_avg' in kwargs else 23.0
    n_ts, n_rg = data.shape[:2]

    # allocate numpy arrays
    mean = np.zeros((n_ts, n_rg), dtype=np.float32)
    thresh = np.zeros((n_ts, n_rg), dtype=np.float32)
    var = np.zeros((n_ts, n_rg), dtype=np.float32)
    edges = np.full((n_ts, n_rg, 2), -222)

    for iT in range(n_ts):
        for iR in range(n_rg):
            mean_, thresh_, var_, nnoise_ = estimate_noise_hs74(data[iT, iR, :], navg=n_avg, std_div=n_std)
            mean[iT, iR] = mean_
            thresh[iT, iR] = thresh_
            var[iT, iR] = var_
            edges[iT, iR, :] = find_peak_edges(data[iT, iR, :], thresh_)
            logger.debug(f'edges(iT={iT}, iR={iR}) = {edges[iT, iR, :]}')
            dummy = 5

    return mean, thresh, var, edges


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

            - Ze_lin (float array): reflectivity (0.Mom) over range of velocity bins [mm6/m3]
            - VEL (float array): mean velocity (1.Mom) over range of velocity bins [m/s]
            - sw (float array):: spectrum width (2.Mom) over range of velocity bins [m/s]
            - skew (float array):: skewness (3.Mom) over range of velocity bins
            - kurt (float array):: kurtosis (4.Mom) over range of velocity bins
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
    """
    SPECKLEFILTER:
        Remove small patches (speckle) from any given mask by checking 5x5 box
        around each pixel, more than half of the points in the box need to be 1
        to keep the 1 at current pixel

    Args:
        mask (numpy.array, integer): 2D mask where 1 = an invalid/fill value and 0 = a data point (time, height)
        min_percentage (float): minimum percentage of neighbours that need to be signal above noise

    Return:
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

    Return:
        container (dict): larda data container
    """

    if len(varname) > 0:
        spectra_all_chirps = [spectra_all_chirps[ic][varname] for ic in range(len(spectra_all_chirps))]

    spectra = spectra_all_chirps[0]

    container = {'dimlabel': ['time', 'range'], 'filename': spectra['filename'], 'paraminfo': copy.deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'], 'colormap': paraminfo['colormap'],
                 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'],
                 'system': paraminfo['system'], 'name': paraminfo['paramkey'],
                 'rg': np.array([rg for ic in spectra_all_chirps for rg in ic['rg']]), 'ts': spectra['ts'],
                 'mask': invalid_mask, 'var': values[:]}

    return container


def load_spectra_rpgfmcw94(larda, time_span, **kwargs):
    """
    This routine will generate a list of larda containers including spectra of the RPG-FMCW 94GHz radar.
    The list-container at return will contain the additional information, for each chirp:
        - spec[i_chirps]['no_av'] (float): Number of spectral averages divided by the number of FFT points
        - spec[i_chirps]['DoppRes'] (float): Doppler resolution for
        - spec[i_chirps]['SL'] (2D-float): Sensitivity limit (dimensions: time, range)
        - spec[i_chirps]['NF'] (string): Noise factor, default = 6.0
        - spec[i_chirps]['rg_offsets'] (list): Indices, where chipr shifts

    Args:
        larda (class larda): Initialized pyLARDA, already connected to a specific campaign
        time_span (list): Starting and ending time point in datetime format.

    Kwargs:
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
    Return:
        container (list): list of larda data container
    """

    rpg_radar = 'LIMRAD94'

    # read limrad94 doppler spectra and caluclate radar moments
    std_above_mean_noise = float(kwargs['noise_factor']) if 'noise_factor' in kwargs else 6.0
    ghost_echo_1 = kwargs['ghost_echo_1'] if 'ghost_echo_1' in kwargs else False
    ghost_echo_2 = kwargs['ghost_echo_2'] if 'ghost_echo_2' in kwargs else False
    do_despeckle2D = kwargs['despeckle2D'] if 'despeckle2D' in kwargs else False
    add_horizontal_channel = True if 'add_horizontal_channel' in kwargs and kwargs['add_horizontal_channel'] else False
    estimate_noise = True if std_above_mean_noise > 0.0 else False

    AvgNum_in = larda.read(rpg_radar, "AvgNum", time_span)
    DoppLen_in = larda.read(rpg_radar, "DoppLen", time_span)
    MaxVel_in = larda.read(rpg_radar, "MaxVel", time_span)
    ChirpFFTSize_in = larda.read(rpg_radar, "ChirpFFTSize", time_span)
    data = {}

    # depending on how much files are loaded, AvgNum and DoppLen are multidimensional list
    if len(AvgNum_in['var'].shape) > 1:
        AvgNum = AvgNum_in['var'][0]
        DoppLen = DoppLen_in['var'][0]
        ChirpFFTSize = ChirpFFTSize_in['var'][0]
        DoppRes = np.divide(2.0 * MaxVel_in['var'][0], DoppLen_in['var'][0])
        MaxVel = MaxVel_in['var'][0]
    else:
        AvgNum = AvgNum_in['var']
        DoppLen = DoppLen_in['var']
        ChirpFFTSize = ChirpFFTSize_in['var']
        DoppRes = np.divide(2.0 * MaxVel_in['var'], DoppLen_in['var'])
        MaxVel = MaxVel_in['var']

    # initialize
    tstart = time.time()

    if add_horizontal_channel:
        data['SLh'] = larda.read(rpg_radar, "SLh", time_span, [0, 'max'])
        data['HSpec'] = larda.read(rpg_radar, 'HSpec', time_span, [0, 'max'])
        data['ReVHSpec'] = larda.read(rpg_radar, 'ImVHSpec', time_span, [0, 'max'])
        data['ImVHSpec'] = larda.read(rpg_radar, 'ReVHSpec', time_span, [0, 'max'])

    data['VHSpec'] = larda.read(rpg_radar, 'VSpec', time_span, [0, 'max'])
    data['NF'] = std_above_mean_noise
    data['no_av'] = np.divide(AvgNum, DoppLen)
    data['DoppRes'] = DoppRes
    data['DoppLen'] = DoppLen
    data['MaxVel'] = MaxVel
    data['ChirpFFTSize'] = ChirpFFTSize
    data['n_ts'], data['n_rg'], data['n_vel'] = data['VHSpec']['var'].shape
    data['n_ch'] = len(MaxVel)
    data['rg_offsets'] = [0]
    data['vel'] = []

    # read spectra and other variables
    for ic in range(len(AvgNum)):
        nrange_ = larda.read(rpg_radar, f'C{ic + 1}Range', time_span)['var']
        if len(nrange_.shape) == 1:
            nrange_ = nrange_.size
        else:
            nrange_ = nrange_.shape[1]
        data['rg_offsets'].append(data['rg_offsets'][ic] + nrange_)
        data['vel'].append(np.linspace(-MaxVel[ic] + (0.5 * DoppRes[ic]), +MaxVel[ic] - (0.5 * DoppRes[ic]), np.max(DoppLen)))

    logger.info(f'Loading spectra, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')
    """
    ####################################################################################################################
    ____ ___  ___  _ ___ _ ____ _  _ ____ _       ___  ____ ____ ___  ____ ____ ____ ____ ____ ____ _ _  _ ____ 
    |__| |  \ |  \ |  |  | |  | |\ | |__| |       |__] |__/ |___ |__] |__/ |  | |    |___ [__  [__  | |\ | | __ 
    |  | |__/ |__/ |  |  | |__| | \| |  | |___    |    |  \ |___ |    |  \ |__| |___ |___ ___] ___] | | \| |__]

    ####################################################################################################################                                                                                                             
    """

    if estimate_noise:
        tstart = time.time()
        data['Vnoise'] = larda.read(rpg_radar, 'VNoisePow', time_span, [0, 'max'])
        if add_horizontal_channel: data['Hnoise'] = larda.read(rpg_radar, 'HNoisePow', time_span, [0, 'max'])

        # initialize arrays
        data['mean'] = np.full((data['n_ts'], data['n_rg']), -999.0)
        data['variance'] = np.full((data['n_ts'], data['n_rg']), -999.0)
        tmp = data['VHSpec']['var'].copy()
        tmp[data['VHSpec']['var'] <= 0.0] = np.nan

        # catch RuntimeWarning: All-NaN slice encountered
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data['thresh'] = np.nanmin(tmp, axis=2)
            data['var_max'] = np.nanmax(tmp, axis=2)

        data['edges'] = np.full((data['n_ts'], data['n_rg'], 2), 0, dtype=int)
        del tmp

        # find all-noise-spectra (aka. fill_value)
        mask = np.all(data['VHSpec']['var'] == -999.0, axis=2)
        data['thresh'][mask] = data['Vnoise']['var'][mask]

        for iT in range(data['n_ts']):
            for iR in range(data['n_rg']):
                if mask[iT, iR]: continue
                data['edges'][iT, iR, :] = find_peak_edges(data['VHSpec']['var'][iT, iR, :], data['thresh'][iT, iR])

        logger.info(f'Loading Noise Level, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    if ghost_echo_1:
        tstart = time.time()
        data['ge1_mask'] = filter_ghost_1(data['VHSpec']['var'], data['VHSpec']['rg'], data['vel'], data['rg_offsets'])
        data['VHSpec']['var'][data['ge1_mask']] = -999.0
        logger.info(f'Precipitation Ghost Filter applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    if ghost_echo_2:
        data['SLv'] = larda.read(rpg_radar, "SLv", time_span, [0, 'max'])
        data['ge2_mask'] = filter_ghost_2(data['VHSpec']['var'], data['VHSpec']['rg'], data['SLv']['var'], data['rg_offsets'][1])
        data['VHSpec']['var'][data['ge2_mask']] = -999.0
        logger.info(f'Curtain-like Ghost Filter applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    if do_despeckle2D:
        tstart = time.time()
        data['dspkl_mask'] = despeckle2D(data['VHSpec']['var'])
        data['VHSpec']['var'][data['dspkl_mask']] = -999.0
        logger.info(f'Despeckle applied, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    return data


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


def filter_ghost_1(data, rg, vel, offset, dBZ_thresh=-20.0, reduce_by=1.5):
    """This function is used to remove certain spectral lines "speckle ghost echoes" from all chirps of RPG FMCW 94GHz cloud radar spectra.
    The speckle occur usually near the maximum unambiguous Doppler velocity.

    Args:
        data (numpy.array): cloud radar Doppler spectra, dimensions: (time, range, velocity), unit: [mm6 mm-3 m s-1]
        rg (numpy.array): range values, unit [m]
        vel (list of numpy.arrays): contains the Doppler velocity values for each chirp, dimension = n_chirps
        offset (list, integer): range indices where the chirp changes takes place, dimension = n_chirps + 1 (starting with 0)
        dBZ_thresh (float): values below will be considered as ghost echo
        reduce_by (float): reduce the maximum unambiguous Doppler velocity by this amount in [m s-1]

    Returns:
        mask (numpy.array, bool): where True = fill_value, and False = signal, dimensions: (time, range, velocity)

    """

    # there must be high levels of reflection/scattering in this region to produce ghost echos
    RG_MIN_, RG_MAX_ = 0.0, 500.0  # range interval
    mask = data <= 0.0

    # check the if high signal occurred in 0m - 500m altitude (indicator for speckle ghost echos above)
    dBZ_max = np.max(data[:, argnearest(rg, RG_MIN_):argnearest(rg, RG_MAX_), :], axis=2)
    ts_to_mask = np.any(dBZ_max >= z2lin(0.0), axis=1)

    signal_min = z2lin(dBZ_thresh)
    n_vel = data.shape[2]

    for iC in range(len(vel)):
        if iC < 1: continue  # exclude first chirp because ghost is hidden under real signal anyway
        idx_max_vel_new = argnearest(vel[iC], vel[iC][-1] - reduce_by)
        for iV in range(n_vel - idx_max_vel_new):
            mask[ts_to_mask, offset[iC]:offset[iC + 1], iV] = data[ts_to_mask, offset[iC]:offset[iC + 1], iV] < signal_min
        for iV in range(idx_max_vel_new, n_vel):
            mask[ts_to_mask, offset[iC]:offset[iC + 1], iV] = data[ts_to_mask, offset[iC]:offset[iC + 1], iV] < signal_min

    return mask


def filter_ghost_2(data, rg, SL, first_offset, dBZ_thresh=-5.0, reduce_by=4.0):
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
    RG_MIN_, RG_MAX_ = 1500.0, 4500.0  # range interval
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


def spectra2moments(Z_spec, paraminfo, **kwargs):
    """
    This routine calculates the radar moments: reflectivity, mean Doppler velocity, spectrum width, skewness and
    kurtosis from the level 0 spectrum files of the 94 GHz RPG cloud radar.

    Args:
        Z_spec (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Return:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

    # initialize variables:
    Z = np.full((Z_spec['n_ts'], Z_spec['n_rg']), np.nan)
    V = np.full((Z_spec['n_ts'], Z_spec['n_rg']), np.nan)
    SW = np.full((Z_spec['n_ts'], Z_spec['n_rg']), np.nan)
    SK = np.full((Z_spec['n_ts'], Z_spec['n_rg']), np.nan)
    K = np.full((Z_spec['n_ts'], Z_spec['n_rg']), np.nan)

    spec_lin = Z_spec['VHSpec']['var'].copy()
    mask = spec_lin <= 0.0
    spec_lin[mask] = 0.0

    # combine the mask for "contains signal" with "signal has more than 1 spectral line"
    mask1 = np.all(mask, axis=2)
    mask2 = Z_spec['edges'][:, :, 1] - Z_spec['edges'][:, :, 0] <= 0
    mask3 = Z_spec['edges'][:, :, 1] - Z_spec['edges'][:, :, 0] >= Z_spec['n_vel']
    mask = mask1 * mask2 * mask3

    for iC in range(Z_spec['n_ch']):
        tstart = time.time()
        for iR in range(Z_spec['rg_offsets'][iC], Z_spec['rg_offsets'][iC + 1]):  # range dimension
            for iT in range(Z_spec['n_ts']):  # time dimension
                if mask[iT, iR]: continue
                lb, rb = Z_spec['edges'][iT, iR, :]
                Z[iT, iR], V[iT, iR], SW[iT, iR], SK[iT, iR], K[iT, iR] = \
                    radar_moment_calculation(spec_lin[iT, iR, lb:rb], Z_spec['vel'][iC][lb:rb], Z_spec['DoppRes'][iC])

        logger.info(f'Chirp {iC + 1} Moments Calculated, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    moments = {'Ze': Z, 'VEL': V, 'sw': SW, 'skew': SK, 'kurt': K}
    # create the mask where invalid values have been encountered
    invalid_mask = np.full((Z_spec['VHSpec']['var'].shape[:2]), True)
    invalid_mask[np.where(Z > 0.0)] = False

    # despeckle the moments
    if 'despeckle' in kwargs and kwargs['despeckle']:
        tstart = time.time()
        # copy and convert from bool to 0 and 1, remove a pixel  if more than 20 neighbours are invalid (5x5 grid)
        new_mask = despeckle(invalid_mask, 80.)
        invalid_mask[new_mask] = True
        logger.info(f'Despeckle done, elapsed time = {seconds_to_fstring(time.time() - tstart)} [min:sec]')

    for mom in moments.keys():
        moments[mom][invalid_mask] = -999.0

    # build larda containers from calculated moments
    container_dict = {mom: make_container_from_spectra([Z_spec], moments[mom], paraminfo[mom], invalid_mask, 'VHSpec') for mom in moments.keys()}

    return container_dict


def spectra2polarimetry(Z_spec, paraminfo, **kwargs):
    """
    This routine calculates the

    Args:
        Z_spec (dict): list containing the dicts for each chrip of RPG-FMCW Doppler cloud radar
        paraminfo (dict): information from params_[campaign].toml for the system LIMRAD94

    Return:
        container_dict (dict): dictionary of larda containers, including larda container for Ze, VEL, sw, skew, kurt

    """

    tstart = time.time()  # initialize variables:

    vspec_lin = Z_spec['VHSpec']['var'].copy()  # - Z_spec[iC]['mean']
    hspec_lin = Z_spec['HSpec']['var'].copy()
    vspec_lin = vspec_lin - hspec_lin
    Revhspec_lin = Z_spec['ReVHSpec']['var'].copy()
    Imvhspec_lin = Z_spec['ImVHSpec']['var'].copy()
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
