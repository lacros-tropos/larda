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


def lin2z(array):
    """linear values to dB (for np.array or single number)"""
    return 10*np.log10(array)


def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10**(array/10.)


def fill_with(array, mask, fill):
    """fill an array where mask is true with fill value"""
    filled = array.copy()
    filled[mask] = fill
    return filled
