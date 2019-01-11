#!/usr/bin/python


import datetime
import numpy as np

def ident(x):
    return x

def get_converter_array(string, **kwargs):
    """colletion of converters that works on arrays
    combines time, range and varconverters (i see no conceptual separation here)
   
    the maskconverter becomes relevant, if the order is no
    time, range, whatever (as in mira spec)

    Returns:
        (varconverter, maskconverter)
    """
    if string == 'since20010101':
        return lambda x: x+dt_to_ts(datetime.datetime(2001,1,1)), ident
    elif string == 'unix':
        return lambda x: x, ident
    elif string == 'since19691231':
        return lambda x: x+dt_to_ts(datetime.datetime(1969,12,31,23)), ident
    elif string == 'beginofday':
        if 'ncD' in kwargs.keys():
            return (lambda h: (h.astype(np.float64)*3600.+\
                float(dt_to_ts(datetime.datetime(kwargs['ncD'].year, 
                                           kwargs['ncD'].month, 
                                           kwargs['ncD'].day)))), 
                    ident)

    elif string == "km2m":
        return lambda x: x*1000., ident
    elif string == "sealevel2range":
        return lambda x: x-kwargs['altitude'], ident
    
    elif string == 'z2lin':
        return z2lin, ident
    elif string == 'lin2z':
        return lin2z, ident
    elif string == 'switchsign':
        return lambda x: -x, ident
    
    elif string == 'transposedimensions':
        return np.transpose, np.transpose
    elif string == "none":
        return ident, ident 
    else:
        raise ValueError("rangeconverter {} not defined".format(string))


def flatten(xs):
    """flatten inhomogeneous deep lists
    e.g. ``[[1,2,3],4,5,[6,[7,8],9],10]``
    """
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result


def since2001_to_dt(s):
    """seconds since 2001-01-01 to datetime"""
    #return (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    return datetime.datetime(2001,1,1) + datetime.timedelta(seconds=s)


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    #return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)


def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values
        value: value to find
    Returns:
        index  
    """
    i = np.searchsorted(array, value)-1
    if not i == array.shape[0]-1 \
            and np.abs(array[i]-value) > np.abs(array[i+1]-value):
        i = i+1
    return i

def nearest(array, pivot):
    """find the nearest value to a given one

    Args:
        array (np.array): sorted array with values
        pivot: value to find
    Returns:
        value with smallest distance
    """
    return min(array, key=lambda x: abs(x - pivot))


def lin2z(array):
    """linear values to dB (for np.array or single number)"""
    return 10*np.ma.log10(array)


def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10**(array/10.)


def fill_with(array, mask, fill):
    """fill an array where mask is true with fill value"""
    filled = array.copy()
    filled[mask] = fill
    return filled

# @jit(nopython=True, fastmath=True)
def estimate_noise_hs74(spectrum, **kwargs):
    """
    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.

    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.

    Parameters
    ----------
    Args:
        spectrum (numpy.ndarray): dimension (n_Dopplerbins,)
            Doppler spectrum in linear units.
        **navg (int) optional: The number of spectral bins over which a moving average has been
            taken. Corresponds to the **p** variable from equation 9 of the
            article.  The default value of 1 is appropiate when no moving
            average has been applied to the spectrum.
        **n_std_diviations (float) optional: threshold = number of standart deviations
            above mean noise floor, defalut: threshold is the value of the first
            non-noise value
    Returns:
        mean (float): Mean of points in the spectrum identified as noise.
        threshold (float): Threshold separating noise from signal.  The point in the spectrum with
            this value or below should be considered as noise, above this value
            signal. It is possible that all points in the spectrum are identified
            as noise.  If a peak is required for moment calculation then the point
            with this value should be considered as signal.
        var (float): Variance of the points in the spectrum identified as noise.
        nnoise (int): Number of noise points in the spectrum.
    """

    navg = kwargs['navg'] if 'navg' in kwargs else 1.0

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

    if 'n_std_diviations' in kwargs:
        threshold = mean + np.sqrt(var) * kwargs['n_std_diviations']
    else:
        threshold = sorted_spectrum[nnoise - 1]

    # boundaries of major peak only
    left_intersec = -1
    right_intersec = -1

    if nnoise < n_spec:
        idxMaxSignal = np.argmax(spectrum)

        for ispec in range(idxMaxSignal, n_spec):
            if spectrum[ispec] <= threshold:
                right_intersec = ispec
                break

        for ispec in range(idxMaxSignal, -1, -1):
            if spectrum[ispec] <= threshold:
                left_intersec = ispec
                break

    #    left_intersec  = -1
    #    right_intersec = -1
    #    if nnoise < n_spec:
    #        for ispec in range(n_spec):
    #            if spectrum[ispec] > threshold:
    #                left_intersec  = ispec-1
    #                break
    #
    #        for ispec in range(n_spec-1, -1, -1):
    #            if spectrum[ispec] > threshold:
    #                right_intersec = ispec+1
    #                break
    return mean, threshold, var, nnoise, left_intersec, right_intersec

#
# @jit(nopython=True, fastmath=True)
def noise_estimation(data, **kwargs):
    """
    Creates a dict containing the noise threshold, mean noise level,
    the variance of the noise, the number of noise values in the spectrum,
    and the boundaries of the main signal peak, if there is one

    Args:
        data (dict): data container, containing data['var'] of dimension (n_time, n_range, n_Doppler_bins)
        **n_std_diviations (float): threshold = number of standart deviations
                                    above mean noise floor, defalut: threshold is the value of the first
                                    non-noise value
    Returns:
        noise_est noise (dict): noise floor estimation for all time and range points
    """

    n_std = kwargs['n_std_diviations'] if 'n_std_diviatinons' in kwargs else 1.0

    n_t = data['ts'].size
    n_r = data['rg'].size

    # allocate numpy arrays
    noise_est = {'mean': np.zeros((n_t, n_r), dtype=np.float32),
                  'threshold': np.zeros((n_t, n_r), dtype=np.float32),
                  'variance': np.zeros((n_t, n_r), dtype=np.float32),
                  'numnoise': np.zeros((n_t, n_r), dtype=np.int32),
                  'bounds': np.zeros((n_t, n_r), dtype=np.int32)}

    # gather noise level etc. for all chirps, range gates and times
    for iR in range(n_r):
        for iT in range(n_t):
            mean, thresh, var, nnoise, left, right = estimate_noise_hs74(data['var'][iT, iR, :],
                                                                         navg=data['no_av'],
                                                                         std_div=n_std)

            noise_est['mean_noise'][iT, iR] = mean
            noise_est['variance'][iT, iR] = var
            noise_est['numnoise'][iT, iR] = nnoise
            noise_est['threshold'][iT, iR] = thresh
            noise_est['integration_bounds'][iT, iR, :] = [left, right]

    return noise_est


def spectra_to_moments(spectra_linear_units, velocity_bins, bounds, DoppRes):
    """
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width, skewness, and kurtosis
    translated from Heike's Matlab function
    determination of radar moments of Doppler spectrum over range of Doppler velocity bins
    Note: Each chirp of LIMRAD94 data has to be provided seperatly because
          chirps have in general different n_Doppler_bins and no_av
    Args:
        spectra_linear_units (float): time-height-FFT points of Doppler spectra ([mm^6 / m^3 ] / (m/s)
        velocity_bins (float): FFTpoint-long spectral velocity bins (m/s)
        bounds (int): integration boundaries (separates signal from noise)
    Results:
        moments (dict): containing:
            Ze              : 0. moment = reflectivity over range of Doppler velocity bins v1 to v2 [mm6/m3]
            mdv             : 1. moment = mean Doppler velocity over range of Doppler velocity bins v1 to v2 [m/s]
            sw              : 2. moment = spectrum width over range of Doppler velocity bins v1 to v2  [m/s]
            skew            : 3. moment = skewness over range of Doppler velocity bins v1 to v2
            kurt            : 4. moment = kurtosis over range of Doppler velocity bins v1 to v2
            pwr_nrm_out     : normalized power (the sum of pwr_norm is 1)
    """

    # contains the dimensionality of the Doppler spectrum, (nTime, nRange, nDopplerbins)
    no_times = spectra_linear_units.shape[0]
    no_ranges = spectra_linear_units.shape[1]
    no_Dbins = velocity_bins.size

    # initialize variables:
    # create empty arrays for output
    Ze = np.full((no_times, no_ranges), np.nan)
    mdv = np.full((no_times, no_ranges), np.nan)
    sw = np.full((no_times, no_ranges), np.nan)
    skew = np.full((no_times, no_ranges), np.nan)
    kurt = np.full((no_times, no_ranges), np.nan)

    # add new list element, mean velocity difference between velocity bins:
    pwr_nrm_out = np.full((no_times, no_ranges, no_Dbins), np.nan)

    for iR in range(no_ranges):  # range dimension
        for iT in range(no_times):  # time dimension

            if bounds[iT, iR, 0] > -1:  # check if signal was detected by estimate_noise routine

                lb = int(bounds[iT, iR, 0])
                ub = int(bounds[iT, iR, 1])

                # hier signal durch 2 teilen .... warum?? weil VH = Vspec+Hspec???? ask Alexander Myagkov why
                signal = spectra_linear_units[iT, iR, lb:ub]  # extract power spectra in chosen range

                #if ic == 0:
                #    if np.sum(signal) < 5.e-6:
                #        hydro_meteor = False
                #    elif np.sum(np.ma.diff(np.ma.masked_less_equal(signal, -999.))) < 5.e-6:
                #        hydro_meteor = False
                #    else:
                #        hydro_meteor =  True
                #else:
                hydro_meteor = True

                velocity_bins_extr = velocity_bins[lb:ub]  # extract velocity bins in chosen Vdop bin range
                signal_sum = np.nansum(signal)  # linear full spectrum Ze [mm^6/m^3], scalar

                if np.isfinite(signal_sum) and hydro_meteor:  # check if Ze_linear is not NaN

                    # Ze_linear = signal_sum
                    Ze_linear = signal_sum / 2.0
                    Ze[iT, iR] = Ze_linear  # copy temporary Ze_linear variable to output variable

                    pwr_nrm = signal / signal_sum  # determine normalized power (NOT normalized by Vdop bins)
                    pwr_nrm_out[iT, iR, lb:ub] = pwr_nrm  # create output matrix of normalized power

                    mdv[iT, iR] = np.nansum(velocity_bins_extr * pwr_nrm)
                    sw[iT, iR] = np.sqrt(
                        np.abs(np.nansum(np.multiply(pwr_nrm, np.square(velocity_bins_extr - mdv[iT, iR])))))
                    skew[iT, iR] = np.nansum(
                        pwr_nrm * np.power(velocity_bins_extr - mdv[iT, iR], 3.0)) / np.power(sw[iT, iR], 3.0)
                    kurt[iT, iR] = np.nansum(
                        pwr_nrm * np.power(velocity_bins_extr - mdv[iT, iR], 4.0)) / np.power(sw[iT, iR], 4.0)

                    mdv[iT, iR] = mdv[iT, iR] - DoppRes / 2.0  # ask Alexander Myagkov why

    Ze = np.ma.masked_invalid(Ze)
    mdv = np.ma.masked_invalid(mdv)
    sw = np.ma.masked_invalid(sw)
    skew = np.ma.masked_invalid(skew)
    kurt = np.ma.masked_invalid(kurt)

    return Ze, mdv, sw, skew, kurt, pwr_nrm_out
