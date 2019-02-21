import numpy as np
from numba import jit
import copy


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

    condition = n_spec - nnoise > 2
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


#
# #@jit(nopython=True, fastmath=True)
def noise_estimation(data, **kwargs):
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

    n_std = kwargs['n_std_deviations'] if 'n_std_deviations' in kwargs else 1.0

    n_t = data['ts'].size
    n_r = data['rg'].size

    # allocate numpy arrays
    noise_est = {'mean': np.zeros((n_t, n_r), dtype=np.float32),
                 'threshold': np.zeros((n_t, n_r), dtype=np.float32),
                 'variance': np.zeros((n_t, n_r), dtype=np.float32),
                 'numnoise': np.zeros((n_t, n_r), dtype=np.int32),
                 'signal': np.full((n_t, n_r), fill_value=False),
                 'bounds': np.full((n_t, n_r, 2), fill_value=None)}

    # gather noise level etc. for all chirps, range gates and times
    for iR in range(n_r):
        for iT in range(n_t):
            mean, thresh, var, nnoise, signal, left, right = \
                estimate_noise_hs74(data['var'][iT, iR, :], navg=data['no_av'], std_div=n_std)

            noise_est['mean'][iT, iR] = mean
            noise_est['variance'][iT, iR] = var
            noise_est['numnoise'][iT, iR] = nnoise
            noise_est['threshold'][iT, iR] = thresh
            noise_est['signal'][iT, iR] = signal
            noise_est['bounds'][iT, iR, :] = [left, right]

    return noise_est


def spectra_to_moments_limrad(spectra_linear_units, velocity_bins, signal_flag, bounds, DoppRes, **kwargs):
    """
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width, skewness, and kurtosis
    translated from Heike's Matlab function
    determination of radar moments of Doppler spectrum over range of Doppler velocity bins

    Note:
        Each chirp of LIMRAD94 data has to be provided seperatly because
        chirps have in general different n_Doppler_bins and no_av

    Args:
        spectra_linear_units (float): dimension (time, height, nFFT points) of Doppler spectra ([mm^6 / m^3 ] / (m/s)
        velocity_bins (float): FFTpoint-long spectral velocity bins (m/s)
        signal (logical): True if signal was detected, false otherwise
        bounds (int): integration boundaries (separates signal from noise)

    Returns:
        dict containing
            - Ze_lin: reflectivity (0.Mom) over range of velocity bins lb to ub [mm6/m3]
            - VEL: mean velocity (1.Mom) over range of velocity bins lb to ub [m/s]
            - sw: spectrum width (2.Mom) over range of velocity bins lb to ub  [m/s]
            - skew: skewness (3.Mom) over range of velocity bins lb to ub
            - kurt: kurtosis (4.Mom) over range of velocity bins lb to ub
    """

    # contains the dimensionality of the Doppler spectrum, (nTime, nRange, nDopplerbins)
    no_times = spectra_linear_units.shape[0]
    no_ranges = spectra_linear_units.shape[1]

    # initialize variables:
    moments = {'Ze_lin': np.full((no_times, no_ranges), np.nan),
               'VEL': np.full((no_times, no_ranges), np.nan),
               'sw': np.full((no_times, no_ranges), np.nan),
               'skew': np.full((no_times, no_ranges), np.nan),
               'kurt': np.full((no_times, no_ranges), np.nan)}

    noise_removed = True if 'thresh' in kwargs else False

    for iR in range(no_ranges):  # range dimension
        for iT in range(no_times):  # time dimension

            if signal_flag[iT, iR]:

                if noise_removed:
                    signal = delete_noise(spectra_linear_units[iT, iR, :], kwargs['thresh'][iT, iR])
                    Ze_lin, VEL, sw, skew, kurt = moment_calculation(signal, velocity_bins, DoppRes)
                else:
                    lb = bounds[iT, iR, 0]
                    ub = bounds[iT, iR, 1]

                    signal = spectra_linear_units[iT, iR, lb:ub]  # extract power spectra in chosen range
                    velocity_bins_extr = velocity_bins[lb:ub]  # extract velocity bins in chosen Vdop bin range

                    Ze_lin, VEL, sw, skew, kurt = moment_calculation(signal, velocity_bins_extr, DoppRes)



            else:
                Ze_lin, VEL, sw, skew, kurt = [np.nan] * 5

            moments['Ze_lin'][iT, iR] = Ze_lin  # copy temporary Ze_linear variable to output variable
            moments['VEL'][iT, iR] = VEL
            moments['sw'][iT, iR] = sw
            moments['skew'][iT, iR] = skew
            moments['kurt'][iT, iR] = kurt

    moments['Ze_lin'] = np.ma.masked_invalid(moments['Ze_lin'])
    moments['VEL'] = np.ma.masked_invalid(moments['VEL'])
    moments['sw'] = np.ma.masked_invalid(moments['sw'])
    moments['skew'] = np.ma.masked_invalid(moments['skew'])
    moments['kurt'] = np.ma.masked_invalid(moments['kurt'])

    return moments


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
    VEL = np.nansum(vel_bins * pwr_nrm)
    vel_diff = vel_bins - VEL
    sw = np.sqrt(np.abs(np.nansum(pwr_nrm * vel_diff ** 2.0)))
    skew = np.nansum(pwr_nrm * vel_diff ** 3.0 / sw ** 3.0)
    kurt = np.nansum(pwr_nrm * vel_diff ** 4.0 / sw ** 4.0)
    VEL = VEL - DoppRes / 2.0

    return Ze_lin, VEL, sw, skew, kurt


#@jit(nopython=True, fastmath=True)
def delete_noise(spectrum, threshold):
    spec_no_noise = copy.deepcopy(spectrum)
    spec_no_noise[spectrum < threshold] = np.nan
    return spec_no_noise


def compare_datasets(lv0, lv1):
    """
    Helper function for displaying the mean difference of calculated moments from RPG software and larda.
    Use only for debugging.
    """
    # Z_norm = 10.0 * np.log10(np.linalg.norm(np.ma.subtract(Ze1, Ze2), ord='fro'))
    # VEL_norm = np.linalg.norm(np.subtract(lv0.mdv, lv1.mdv), ord='fro')
    # sw_norm  = np.linalg.norm(np.subtract(lv0.sw, lv1.sw), ord='fro')

    Z_norm = np.mean(np.ma.subtract(lv0['Ze'], lv1['Ze']))
    VEL_norm = np.mean(np.subtract(lv0['VEL'], lv1['VEL']))
    sw_norm = np.mean(np.subtract(lv0['sw'], lv1['sw']))

    # convert to dBZ
    print()
    print('    ||Ze_lv0  -  Ze_lv1|| = {:.12f} [mm6/m3]'.format(Z_norm))
    print('    ||mdv_lv0 - mdv_lv1|| = {:.12f} [m/s]'.format(VEL_norm))
    print('    ||sw_lv0  -  sw_lv1|| = {:.12f} [m/s]'.format(sw_norm))
    print()

    pass
