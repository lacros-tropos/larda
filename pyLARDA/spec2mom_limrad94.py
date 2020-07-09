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

sys.path.append('../../larda/')
sys.path.append('.')

import numpy as np
from numba import jit
import copy, time
import pyLARDA.helpers as h

from datetime import timedelta



import numpy as np

@jit(nopython=True, fastmath=True)
def estimate_noise_hs74_fast(spectrum, navg=1, std_div=6.0, nnoise_min=1):
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
        signal_flag (bool): True if spectrum contains a signal
        left_intersec (int): index of intersection of signal and threshold (left side)
        left_intersec (int): index of intersection of signal and threshold (right side)
    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.
    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise
    n_spec = nnoise

    rtest = 1+1/navg
    sum1 = 0.
    sum2 = 0.
    for i, pwr in enumerate(sorted_spectrum):
        npts = i+1
        if npts < nnoise_min:
            continue

        sum1 += pwr
        sum2 += pwr*pwr

        if npts*sum2 < sum1*sum1*rtest:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise.
            sum1 -= pwr
            sum2 -= pwr*pwr
            break

    mean = sum1/nnoise
    var = sum2/nnoise-mean*mean

    threshold = mean + np.sqrt(var) * std_div

    #threshold = sorted_spectrum[nnoise-1]

    # boundaries of major peak only
    left_intersec = -111
    right_intersec = -111

    # save only signal if it as more than 2 points above the noise threshold
    idxMaxSignal = np.argmax(spectrum)
    signal_flag = True

    for ispec in range(idxMaxSignal, n_spec):
        if spectrum[ispec] <= threshold:
            right_intersec = ispec
            break
        else:
            right_intersec = -222  # strong signal till max Nyquist Velocity, maybe folding?

    for ispec in range(idxMaxSignal, -1, -1):
        if spectrum[ispec] <= threshold:
            left_intersec = ispec
            break
        else:
            left_intersec = -222  # strong signal till min Nyquist Velocity, maybe folding?

    return mean, threshold, var, nnoise, signal_flag, left_intersec, right_intersec

@jit(nopython=True, fastmath=True)
def estimate_noise_hs74(spectrum, navg=1.0, std_div=-1.0):
    """
    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.

    References:
        P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
        Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13, 808-811.

    Args:
        spectrum (numpy.ndarray): dimension (n_Dopplerbins,) Doppler spectrum in linear units.
        **navg (int, optional): The number of spectral bins over which a moving average has been
            taken. Corresponds to the p variable from equation 9 of the
            article.  The default value of 1 is appropriate when no moving
            average has been applied to the spectrum.
        **std_div (float, optional): threshold = number of standard deviations
            above mean noise floor, default threshold is the value of the first
            non-noise value

    Returns:
        list containing

            - mean (float): Mean of points in the spectrum identified as noise.
            - threshold (float): Threshold separating noise from signal. The point in the spectrum with
              this value or below should be considered as noise, above this value
              signal. It is possible that all points in the spectrum are identified
              as noise.  If a peak is required for moment calculation then the point
              with this value should be considered as signal.
            - var (float): Variance of the points in the spectrum identified as noise.
            - nnoise (int): Number of noise points in the spectrum.
    """

    n_spec = len(spectrum)
    sorted_spectrum = np.sort(spectrum)
    nnoise = n_spec  # default to all points in the spectrum as noise
    for npts in range(1, n_spec + 1):
        partial = sorted_spectrum[:npts]
        mean = np.mean(partial)
        var = np.var(partial)
        if var * navg < mean * mean:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise
            break

    noise_spectrum = sorted_spectrum[:nnoise]
    mean = np.mean(noise_spectrum)
    var = np.var(noise_spectrum)

    threshold = mean + np.sqrt(var) * std_div

    # boundaries of major peak only
    left_intersec = -111
    right_intersec = -111
    signal_flag = False

    # save only signal if it as more than 2 points above the noise threshold
    # condition = n_spec - nnoise > 0
    condition = True
    # condition = nnoise < n_spec:

    if condition:
        idxMaxSignal = np.argmax(spectrum)
        signal_flag = True

        for ispec in range(idxMaxSignal, n_spec):
            if spectrum[ispec] <= threshold:
                right_intersec = ispec
                break
            else:
                right_intersec = -222  # strong signal till max Nyquist Velocity, maybe folding?

        for ispec in range(idxMaxSignal, -1, -1):
            if spectrum[ispec] <= threshold:
                left_intersec = ispec
                break
            else:
                left_intersec = -222  # strong signal till min Nyquist Velocity, maybe folding?

    return mean, threshold, var, nnoise, signal_flag, left_intersec, right_intersec


@jit(nopython=True)
def find_main_peak(signal):
    idxMaxSignal = np.argmax(signal)
    thresh = np.min(signal)
    len_sig = len(signal)

    for ispec in range(idxMaxSignal, len_sig):
        if signal[ispec] < thresh:
            right_intersec = ispec
            break
        else:
            # strong signal till max Nyquist Velocity, maybe folding?
            right_intersec = -222

    for ispec in range(idxMaxSignal, -1, -1):
        if signal[ispec] < thresh:
            left_intersec = ispec
            # strong signal till max Nyquist Velocity, maybe folding?
            break
        else:
            left_intersec = -222

    return thresh, left_intersec, right_intersec


#
def noise_estimation(data, **kwargs):
    """
    Creates a dict containing the noise threshold, mean noise level,
    the variance of the noise, the number of noise values in the spectrum,
    and the boundaries of the main signal peak, if there is one

    Args:
        data (list): list of data container, containing data[ichirp]['var'] of dimension (n_ts, n_range, n_Doppler_bins)
        **n_std_deviations (float): threshold = number of standard deviations
                                    above mean noise floor, default: threshold is the value of the first
                                    non-noise value

    Returns:
        dict with noise floor estimation for all time and range points
    """

    n_std = kwargs['n_std_deviations'] if 'n_std_deviations' in kwargs else 1.0
    include_noise = kwargs['include_noise'] if 'include_noise' in kwargs else False
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else False

    n_chirps = len(data)
    n_dbin = data[0]['var'].shape[2] // 2

    # fill values needs to be masked for noise removal otherwise wrong results
    for ic in range(n_chirps):
        if -999.0 in data[ic]['var']:
            data[ic]['var'][data[ic]['var'] == -999.0] = np.nan

    # if one wants to calculate the moments including the noise
    noise_est = []
    if not include_noise:
        # Estimate Noise Floor for all chirps, time stemps and range gates aka. for all pixels
        # Algorithm used: Hildebrand & Sekhon
        for ic in range(n_chirps):

            n_t = data[ic]['ts'].size
            n_r = data[ic]['rg'].size
            tstart = time.time()

            # allocate numpy arrays
            noise_est.append({'mean': np.zeros((n_t, n_r), dtype=np.float32),
                              'threshold': np.zeros((n_t, n_r), dtype=np.float32),
                              'variance': np.zeros((n_t, n_r), dtype=np.float32),
                              'numnoise': np.zeros((n_t, n_r), dtype=np.int32),
                              'signal': np.full((n_t, n_r), fill_value=True),
                              'bounds': np.full((n_t, n_r, 2), fill_value=None)})

            # gather noise level etc. for all chirps, range gates and times
            for iT in range(n_t):
                # it is ok to check only the first range gate here, because every signal contains at least one value below the noise floor
                nonoise = any(np.isnan(data[ic]['var'][iT, 0, :]))
                for iR in range(n_r):
                    if not nonoise:
                        mean, thresh, var, nnoise, signal, left, right = \
                            estimate_noise_hs74_fast(data[ic]['var'][iT, iR, :], navg=data[ic]['no_av'], std_div=n_std)

                        noise_est[ic]['mean'][iT, iR] = mean
                        noise_est[ic]['variance'][iT, iR] = var
                        noise_est[ic]['numnoise'][iT, iR] = nnoise
                        noise_est[ic]['threshold'][iT, iR] = thresh
                        noise_est[ic]['signal'][iT, iR] = signal
                        noise_est[ic]['bounds'][iT, iR, :] = [left, right]
                        #data[ic]['var'][iT, iR, data[ic]['var'][iT, iR, :] < thresh] = -999.0


                    elif main_peak and not nonoise:

                        # filter for c1 ghost echos?
                        # sum_SaN_neg = np.count_nonzero(~np.isnan(data[ic]['var'][iT, iR, :n_dbin]))
                        # sum_SaN_pos = np.count_nonzero(~np.isnan(data[ic]['var'][iT, iR, n_dbin:]))
                        # if sum_SaN_neg < 10 and sum_SaN_pos < 8 and ic == 0:
                        #    thresh, left, right = np.min(data[ic]['var'][iT, iR, :]), -222, -222
                        #    data[ic]['var'][iT, iR, :] = np.nan
                        # else:
                        thresh, left, right = find_main_peak(data[ic]['var'][iT, iR, :])

                        noise_est[ic]['threshold'][iT, iR] = thresh
                        noise_est[ic]['bounds'][iT, iR, :] = [left, right]

            noise_est[ic]['bounds'][noise_est[ic]['bounds'] == -222] = None
            print('noise removed, chirp = {}, elapsed time = {:.3f} sec.'.format(ic + 1, time.time() - tstart))

    return noise_est


def spectra_to_moments_rpgfmcw94(spectrum_container, **kwargs):
    """
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width, skewness, and kurtosis
    translated from Heike's Matlab function
    determination of radar moments of Doppler spectrum over range of Doppler velocity bins

    Note:
        Each chirp of LIMRAD94 data has to be provided separately because
        chirps have in general different n_Doppler_bins and no_av

    Args:
        - spectrum_container (dict): container including VSpec of rpg radar + other variables
        - noise_est (dict): container including mean noise level, noise threshold, ... from noise_estimation routine
        - **include_noise (logical): calculate moments without removing the noise

    Returns:
        dict containing
            - Ze_lin: reflectivity (0.Mom) over range of velocity bins lb to ub [mm6/m3]
            - VEL: mean velocity (1.Mom) over range of velocity bins lb to ub [m/s]
            - sw: spectrum width (2.Mom) over range of velocity bins lb to ub  [m/s]
            - skew: skewness (3.Mom) over range of velocity bins lb to ub
            - kurt: kurtosis (4.Mom) over range of velocity bins lb to ub
    """

    # contains the dimensionality of the Doppler spectrum, (nTime, nRange, nDopplerbins)
    noise_params = ['mean', 'threshold', 'variance', 'signal', 'numnoise', 'bounds']
    n_chirps = len(spectrum_container)
    no_times = spectrum_container[0]['ts'].size
    cum_rg = [0]
    noise_est = []
    no_ranges_tot = 0
    for ic, ichirp in enumerate(spectrum_container):
        cum_rg.append(cum_rg[ic] + ichirp['rg'].size)
        noise_est.append({ivar: spectrum_container[ic][ivar] for ivar in noise_params})
        no_ranges_tot += ichirp['rg'].size

    include_noise = kwargs['include_noise'] if 'include_noise' in kwargs else False
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else False

    # initialize variables:
    moments = {'Ze': np.full((no_ranges_tot, no_times), np.nan),
               'VEL': np.full((no_ranges_tot, no_times), np.nan),
               'sw': np.full((no_ranges_tot, no_times), np.nan),
               'skew': np.full((no_ranges_tot, no_times), np.nan),
               'kurt': np.full((no_ranges_tot, no_times), np.nan),
               'mask': np.full((no_ranges_tot, no_times), True)}

    for ic in range(n_chirps):
        tstart = time.time()
        no_ranges = spectrum_container[ic]['rg'].size

        spectra_linear_units = spectrum_container[ic]['var']
        velocity_bins = spectrum_container[ic]['vel']
        DoppRes = spectrum_container[ic]['DoppRes']

        # if ic == 3:
        #    mask = spectra_linear_units < h.lin2z(25.0) and
        #    spectra_linear_units[] = spectra_linear_units

        for iR in range(no_ranges):  # range dimension
            for iT in range(no_times):  # time dimension
                if include_noise:
                    # type1: calculate moments just for main peak, bounded by 2 values lb and ub
                    signal = spectra_linear_units[iT, iR, :]  # extract power spectra in chosen range
                    velocity_bins_extr = velocity_bins  # extract velocity bins in chosen Vdop bin range
                    Ze_lin, VEL, sw, skew, kurt = moment_calculation(signal, velocity_bins_extr, DoppRes)

                else:
                    Ze_lin, VEL, sw, skew, kurt = [np.nan] * 5

                    if noise_est[ic]['signal'][iT, iR]:
                        if main_peak:
                            lb, rb = noise_est[ic]['bounds'][iT, iR, :]
                            spec_no_noise = copy.deepcopy(spectra_linear_units[iT, iR, lb:rb])
                            velocity_bins_extr = velocity_bins[lb:rb]
                            Ze_lin, VEL, sw, skew, kurt = moment_calculation(spec_no_noise, velocity_bins_extr, DoppRes)
                        else:
                            spec_no_noise = copy.deepcopy(spectra_linear_units[iT, iR, :])
                            spec_no_noise[spectra_linear_units[iT, iR, :] < noise_est[ic]['threshold'][iT, iR]] = np.nan
                            Ze_lin, VEL, sw, skew, kurt = moment_calculation(spec_no_noise, velocity_bins, DoppRes)

                iR_tot = cum_rg[ic] + iR

                moments['Ze'][iR_tot, iT] = Ze_lin  # copy temporary Ze_linear variable to output variable
                moments['VEL'][iR_tot, iT] = VEL
                moments['sw'][iR_tot, iT] = sw
                moments['skew'][iR_tot, iT] = skew
                moments['kurt'][iR_tot, iT] = kurt

        print('moments calculated, chrip = {}, elapsed time = {:.3f} sec.'.format(ic + 1, time.time() - tstart))

    # mask all invalid values (NaN)
    for imom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        moments[imom] = np.ma.masked_invalid(moments[imom])

    # mask values <= 0.0
    moments['Ze'] = np.ma.masked_less_equal(moments['Ze'], 0.0)

    # create the mask where invalid values have been encountered
    moments['mask'][np.where(moments['Ze'] > 0.0)] = False

    return moments


@jit(nopython=True)
def check_signal(spec, thresh):
    """
    This helper function checks if a spectrum contains num_cons consecutive values above the noise threshold.

    Args:
        - spec (numpy array, float): contains one spectrum

    Return:
        - sig (logical): True if spectrum contains num_cons consecutive values above the noise threshold, False otherwise.
    """
    num_cons = 6

    for i in range(len(spec) - num_cons - 1):
        sig = True
        for val in spec[i:i + num_cons]:
            if val < thresh or np.isnan(val):
                sig = False
                break

        if sig:
            return True

    return False


@jit(nopython=True, fastmath=True)
def moment_calculation(signal, vel_bins, DoppRes):
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

    signal_sum = np.nansum(signal)  # linear full spectrum Ze [mm^6/m^3], scalar
    Ze_lin = signal_sum / 2.0
    pwr_nrm = signal / signal_sum  # determine normalized power (NOT normalized by Vdop bins)

    VEL, sw, skew, kurt = [np.nan] * 4
    if not signal_sum == 0.0:
        VEL = np.nansum(vel_bins * pwr_nrm)
        vel_diff = vel_bins - VEL
        vel_diff2 = vel_diff*vel_diff
        sw = np.sqrt(np.abs(np.nansum(pwr_nrm * vel_diff2)))
        sw2 = sw*sw
        skew = np.nansum(pwr_nrm * vel_diff  * vel_diff2 / (sw * sw2))
        kurt = np.nansum(pwr_nrm * vel_diff2 * vel_diff2 / (sw2 * sw2))
        VEL = VEL - DoppRes / 2.0

    return Ze_lin, VEL, sw, skew, kurt


def filter_ghost_echos_RPG94GHz_FMCW(data, **kwargs):
    ######################################################################
    #
    # 2nd and 3rd chirp ghost echo filter
    if 'clean_spectra' in kwargs and kwargs['clean_spectra']:

        sensitivity_limit = kwargs['SL'] if 'SL' in kwargs else sys.exit(
            'Error in clean_spectra ghost echo filter :: Sensitivity Limit missing!')
        Ze = kwargs['Ze'] if 'Ze' in kwargs else sys.exit(
            'Error in clean_spectra ghost echo filter :: Ze values missing!')

        for ichirp in [0, 1, 2]:

            # threholds for 3rd chrip ghost echo filter
            new_ny_vel = data[ichirp]['vel'].max() - 2.5

            ic_Ze_max = h.z2lin(-22.5)

            idx_left = np.argwhere(-new_ny_vel > data[ichirp]['vel']).max()
            idx_right = np.argwhere(new_ny_vel < data[ichirp]['vel']).min()

            Ze_lin_left = data[ichirp]['var'][:, :, :idx_left].copy()
            Ze_lin_right = data[ichirp]['var'][:, :, idx_right:].copy()

            # if noise was already removed by the RPG software, replace the ghost with -999.,
            # if noise factor 0 was selected in the RPG software, replace the ghost by the minimum spectrum value,
            # to avoid wrong noise estimations (to much signal would be lost otherwise),
            idx_ts_nf0 = np.argwhere(data[ichirp]['var'][:, 0, 0] != -999.0)

            if idx_ts_nf0.size > 0:
                mask_left, mask_right = Ze_lin_left < ic_Ze_max, Ze_lin_right < ic_Ze_max
                min_left, min_right = np.amin(Ze_lin_left, axis=2), np.amin(Ze_lin_right, axis=2)

                for i_bin in range(mask_left.shape[2]):
                    Ze_lin_left[mask_left[:, :, i_bin], i_bin] = min_left[mask_left[:, :, i_bin]]
                for i_bin in range(mask_right.shape[2]):
                    Ze_lin_right[mask_right[:, :, i_bin], i_bin] = min_right[mask_right[:, :, i_bin]]
            else:
                Ze_lin_left[Ze_lin_left < ic_Ze_max] = -999.0
                Ze_lin_right[Ze_lin_right < ic_Ze_max] = -999.0

            data[ichirp]['var'][:, :, :idx_left] = Ze_lin_left.copy()
            data[ichirp]['var'][:, :, idx_right:] = Ze_lin_right.copy()
    ######################################################################
    #
    # 2nd and 3rd chirp ghost echo filter
    if 'C2C3' in kwargs and kwargs['C2C3']:
        for ichirp in [0, 1, 2]:

            # threholds for 3rd chrip ghost echo filter
            new_ny_vel = data[ichirp]['vel'].max() - 2.5

            ic_Ze_max = h.z2lin(-22.5)

            idx_left = np.argwhere(-new_ny_vel > data[ichirp]['vel']).max()
            idx_right = np.argwhere(new_ny_vel < data[ichirp]['vel']).min()

            Ze_lin_left = data[ichirp]['var'][:, :, :idx_left].copy()
            Ze_lin_right = data[ichirp]['var'][:, :, idx_right:].copy()

            # if noise was already removed by the RPG software, replace the ghost with -999.,
            # if noise factor 0 was selected in the RPG software, replace the ghost by the minimum spectrum value,
            # to avoid wrong noise estimations (to much signal would be lost otherwise),
            idx_ts_nf0 = np.argwhere(data[ichirp]['var'][:, 0, 0] != -999.0)

            if idx_ts_nf0.size > 0:
                mask_left, mask_right = Ze_lin_left < ic_Ze_max, Ze_lin_right < ic_Ze_max
                min_left, min_right = np.amin(Ze_lin_left, axis=2), np.amin(Ze_lin_right, axis=2)

                for i_bin in range(mask_left.shape[2]):
                    Ze_lin_left[mask_left[:, :, i_bin], i_bin] = min_left[mask_left[:, :, i_bin]]
                for i_bin in range(mask_right.shape[2]):
                    Ze_lin_right[mask_right[:, :, i_bin], i_bin] = min_right[mask_right[:, :, i_bin]]
            else:
                Ze_lin_left[Ze_lin_left < ic_Ze_max] = -999.0
                Ze_lin_right[Ze_lin_right < ic_Ze_max] = -999.0

            data[ichirp]['var'][:, :, :idx_left] = Ze_lin_left.copy()
            data[ichirp]['var'][:, :, idx_right:] = Ze_lin_right.copy()

    ######################################################################
    #
    # 1st chirp ghost echo filter
    if 'C1' in kwargs and kwargs['C1']:
        # invalid mask has to be provided via kwarg for this filter
        invalid_mask = kwargs['inv_mask']
        rg_offsets = kwargs['offset']
        SL = kwargs['SL']

        # setting higher threshold if chirp 2 contains high reflectivity values
        # sum_over_heightC1 = np.ma.sum(data['Ze'][:rg_offsets[1], :], axis=0)
        sum_over_heightC2 = np.ma.sum(data['Ze'][rg_offsets[1]:rg_offsets[2], :], axis=0)

        # load sensitivity limits (time, height) and calculate the mean over time
        sens_reduction = 15.0  # sensitivity in chirp 1 is reduced by 12.0 dBZ
        C2_Ze_threshold = 18.0  # if sum(C2_Ze) > this threshold, ghost echo in C1 is assumed

        sens_lim = h.z2lin(h.lin2z(np.mean(SL, axis=0)) + sens_reduction)
        ts_to_mask = np.argwhere(h.lin2z(sum_over_heightC2) > C2_Ze_threshold)[:, 0]

        m1 = invalid_mask[:rg_offsets[1], :].copy()
        for idx_ts in ts_to_mask:
            m1[:, idx_ts] = data['Ze'][:rg_offsets[1], idx_ts] < sens_lim
            # a = data['Ze'][:rg_offsets[1], idx_ts] < sens_lim
            # b = data['VEL'][:rg_offsets[1], idx_ts] < -0.5
            # m1[:, idx_ts] = a == b

        invalid_mask[:rg_offsets[1], :] = m1.copy()

        for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
            data[mom][:rg_offsets[1], :] = np.ma.masked_where(m1, data[mom][:rg_offsets[1], :])

        return invalid_mask


@jit(nopython=True, fastmath=True)
def despeckle(mask, min_percentage):
    """
    SPECKLEFILTER:
        Remove small patches (speckle) from any given mask by checking 5x5 box
        around each pixel, more than half of the points in the box need to be 1
        to keep the 1 at current pixel

    Args:
        mask (numpy.array, integer): mask where 1 = an invalid/fill value and 0 = a data point [height x time]
        min_percentage (float): minimum percentage of neighbours that need to be signal above noise

    Return:
        mask ... speckle-filtered matrix of 0 and 1 that represents (cloud) mask [height x time]

    """

    WSIZE = 5  # 5x5 window
    n_bins = WSIZE*WSIZE
    min_bins = int(min_percentage/100 * n_bins)
    shift = int(WSIZE / 2)
    n_rg, n_ts = mask.shape

    for iR in range(n_rg - WSIZE):
        for iT in range(n_ts - WSIZE):
            if mask[iR, iT] == 1 and np.sum(mask[iR:iR + WSIZE, iT:iT + WSIZE]) > min_bins:
                mask[iR + shift, iT + shift] = 1

    return mask

@jit(nopython=True, fastmath=True)
def despeckle3d(mask, min_percentage):
    """
    SPECKLEFILTER

    last modification: Heike Kalesse, April 26, 2017; kalesse@tropos.de

    TBD:
    - define percentage of neighboring points that have to be 1 in order to keep pxl value as 1 (instead of "hard" number)
    - what is "C" (in output?)

    functionality:
        remove small patches (speckle) from any given mask by checking 5x5 box
        around each pixel, more than half of the points in the box need to be 1
        to keep the 1 at current pixel

    Args:
        mask         ... mask where 1 = an invalid/fill value and 0 = a data point [height x time]
        nr_neighbors ... number of neighbors of pixel that have to be 1 in order to keep pixel value as 1


    Return:
        mask2 ... speckle-filtered matrix of 0 and 1 that represents (cloud) mask [height x time]

    example of a proggi using this function:
    % % % filter out speckles of liq (this is done later in the Shupe 2007 algorithm)
    % % nr_neighbors = 15;  % number of neighbors of pixel (in a 5x5 matrix; i.e., 25pxl) that have to be 1 in order to keep pixel value as 1 in "speckleFilter.m" (orig=12)
    % % [liq_mask,C] = speckleFilter(liq_mask, nr_neighbors);
    20 neighbors in 5x5 matrix means 80%
    """

    window_size = 5  # 5x5x5 window
    n_bins = window_size*window_size*window_size
    min_bins = int(min_percentage/100 * n_bins)
    shift = int(window_size / 2)
    n_rg, n_ts, n_vel = np.array(mask.shape) - window_size

    for iR in range(n_rg):
        for iT in range(n_ts):
            for iB in range(n_vel):
                if mask[iR, iT, iB] == 1:
                    # selecting window of 5x5 pixels
                    window = mask[iR:iR + window_size, iT:iT + window_size, iB:iB + window_size]

                    # if more than n_neighbours pixel in the window are fill values, remove the pixel in the middle
                    if np.sum(window) > min_bins:
                        mask[iR + shift, iT + shift, iB + shift] = 1

    return mask

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

    spectra = spectra_all_chirps[0]

    container = {'dimlabel': ['time', 'range'], 'filename': spectra['filename'], 'paraminfo': copy.deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'], 'colormap': paraminfo['colormap'],
                 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'],
                 'system': paraminfo['system'], 'name': paraminfo['paramkey'],
                 'rg': np.array([rg for ic in spectra_all_chirps for rg in ic['rg']]), 'ts': spectra['ts'],
                 'mask': invalid_mask, 'var': values[:]}

    return container


def build_extended_container(larda, spectra_ch, time_span, **kwargs):
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
        spectra_ch (string): variable name of the spectra to load (in general either "VSpec" or "HSpec")
        time_span (list): Starting and ending time point in datetime format.

    Kwargs:
        **noise_factor (float): Noise factor, number of standard deviations from mean noise floor
        **rm_precip_ghost (bool): Filters ghost echos which occur over all chirps during precipitation.
        **despeckle3d (float): Removes a pixel in "var" if sounding pixels (5x5x5 box) contain at least despeck% nonzeros.
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

    # read limrad94 doppler spectra and caluclate radar moments
    std_above_mean_noise = float(kwargs['noise_factor']) if 'noise_factor'    in kwargs else 6.0
    rm_precip_ghost      = kwargs['rm_precip_ghost']     if 'rm_precip_ghost' in kwargs else False
    despeckle3d_perc     = kwargs['do_despeckle3d']      if 'do_despeckle3d'  in kwargs else 80.
    estimate_noise       = kwargs['estimate_noise']      if 'estimate_noise'  in kwargs else False

    AvgNum_in  = larda.read("LIMRAD94", "AvgNum", time_span)
    DoppLen_in = larda.read("LIMRAD94", "DoppLen", time_span)
    MaxVel_in  = larda.read("LIMRAD94", "MaxVel", time_span)

    if spectra_ch[0] == 'H':
        SensitivityLimit = larda.read("LIMRAD94", "SLh", time_span, [0, 'max'])
    else:
        SensitivityLimit = larda.read("LIMRAD94", "SLv", time_span, [0, 'max'])

    # depending on how much files are loaded, AvgNum and DoppLen are multidimensional list
    if len(AvgNum_in['var'].shape) > 1:
        AvgNum = AvgNum_in['var'][0]
        DoppLen = DoppLen_in['var'][0]
        DoppRes = np.divide(2.0 * MaxVel_in['var'][0], DoppLen_in['var'][0])
    else:
        AvgNum = AvgNum_in['var']
        DoppLen = DoppLen_in['var']
        DoppRes = np.divide(2.0 * MaxVel_in['var'], DoppLen_in['var'])

    #  dimensions:
    #       -   LIMRAD_Zspec[:]['var']      [Nchirps][ntime, nrange]
    #       -   LIMRAD_Zspec[:]['vel']      [Nchirps][nDoppBins]
    #       -   LIMRAD_Zspec[:]['no_av']    [Nchirps]
    #       -   LIMRAD_Zspec[:]['DoppRes']  [Nchirps]
    rg_offsets = [0]
    spec = []
    for ic in range(len(AvgNum)):
        tstart = time.time()
        var_string = 'C{}{}'.format(ic + 1, spectra_ch)
        spec.append(larda.read("LIMRAD94", var_string, time_span, [0, 'max']))
        ic_n_ts, ic_n_rg, ic_n_nfft = spec[ic]['var'].shape
        rg_offsets.append(rg_offsets[ic] + ic_n_rg)
        spec[ic].update({'no_av': np.divide(AvgNum[ic], DoppLen[ic]),
                         'DoppRes': DoppRes[ic],
                         'SL': SensitivityLimit['var'][:, rg_offsets[ic]:rg_offsets[ic + 1]],
                         'NF': std_above_mean_noise})
        print('reading C{}{}, elapsed time = {} [hrs:min:sec].'.format(
            ic+1, spectra_ch, timedelta(seconds=int(time.time() - tstart))
        ))

    for ic in range(len(AvgNum)):
        spec[ic]['rg_offsets'] = rg_offsets

    """
    ####################################################################################################################
    ____ ___  ___  _ ___ _ ____ _  _ ____ _       ___  ____ ____ ___  ____ ____ ____ ____ ____ ____ _ _  _ ____ 
    |__| |  \ |  \ |  |  | |  | |\ | |__| |       |__] |__/ |___ |__] |__/ |  | |    |___ [__  [__  | |\ | | __ 
    |  | |__/ |__/ |  |  | |__| | \| |  | |___    |    |  \ |___ |    |  \ |__| |___ |___ ___] ___] | | \| |__]
    
    ####################################################################################################################                                                                                                             
    """

    # 3rd chirp ghost echo filter
    if rm_precip_ghost:
        tstart = time.time()
        filter_ghost_echos_RPG94GHz_FMCW(spec, C2C3=True)
        print('precipitation ghost filter done, elapsed time = {} [hrs:min:sec].'.format(
            timedelta(seconds=int(time.time() - tstart))
        ))

    # despeckle the spectra
    if despeckle3d_perc > 0.0:
        # copy and convert from bool to 0 and 1, remove a pixel if more than 80% of neighbours are invalid (5x5x5 grid)
        for ic in range(len(AvgNum)):
            t0, n_inv0 = time.time(), np.sum(spec[ic]['mask'])
            speckle_mask = despeckle3d(spec[ic]['mask'] * 1, despeckle3d_perc) == 1
            spec[ic]['mask'][speckle_mask] = True
            spec[ic]['var'][speckle_mask]  = -999.0
            t1, n_inv1 = timedelta(seconds=int(time.time() - t0)), np.sum(spec[ic]['mask'])
            print('despeckle3d added {} to {} invalid pixel of the spectra in chirp {}, elapsed time = {} [hrs:min:sec].'.format(
                n_inv1-n_inv0, n_inv0, ic+1, t1
            ))

    # noise estimation a la Hildebrand & Sekhon
    if estimate_noise:
        # Logicals for different tasks
        include_noise = True if spec[0]['NF'] < 0.0 else False
        main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True

        # remove noise from raw spectra and calculate radar moments
        # dimensions:
        #       -   noise_est[ichirp]['mean']        [Nchirps][ntime, nrange]
        #       -   noise_est[ichirp]['threshold']   [Nchirps][ntime, nrange]
        #       -   noise_est[ichirp]['variance']    [Nchirps][ntime, nrange]
        #       -   noise_est[ichirp]['numnoise']    [Nchirps][ntime, nrange]
        #       -   noise_est[ichirp]['signal']      [Nchirps][ntime, nrange]
        #       -   noise_est[ichirp]['bounds']      [Nchirps][ntime, nrange, 2]
        noise_est = noise_estimation(spec, n_std_deviations=spec[0]['NF'], include_noise=include_noise, main_peak=main_peak)

        # save noise estimation (mean noise, noise threshold to spectra dict
        for iC in range(len(spec)):
            spec[iC].update({key: noise_est[iC][key].copy() for key in noise_est[iC].keys()})

    return spec


def spectra2moments(Z_spec, paraminfo, **kwargs):
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
    # moment calculation
    include_noise = True if Z_spec[0]['NF'] < 0.0 else False
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True
    moments = spectra_to_moments_rpgfmcw94(Z_spec, include_noise=include_noise, main_peak=main_peak)
    invalid_mask = moments['mask'].copy()

    ####################################################################################################################
    #
    # 1st chirp ghost echo filter
    # test differential phase filter technique
    if 'filter_ghost_C1' in kwargs and kwargs['filter_ghost_C1']:
        invalid_mask = filter_ghost_echos_RPG94GHz_FMCW(moments,
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
        new_mask = despeckle(invalid_mask.copy() * 1, 80.)
        invalid_mask[new_mask == 1] = True

    for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']:
        moments[mom][invalid_mask] = -999.0

        print('despeckle done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    ####################################################################################################################
    #
    # build larda containers from calculated moments
    container_dict = {mom: make_container_from_spectra(Z_spec, moments[mom].T, paraminfo[mom], invalid_mask.T) for mom in ['Ze', 'VEL', 'sw', 'skew', 'kurt']}

    return container_dict
