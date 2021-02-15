#!/usr/bin/python


import datetime, os, copy
import numpy as np
import pprint as pp
import re
import errno
import ast
import traceback

from functools import reduce

import logging

logger = logging.getLogger(__name__)


def ident(x):
    return x


def get_converter_array(string, **kwargs):
    """colletion of converters that works on arrays
    combines time, range and varconverters (i see no conceptual separation here)
   
    the maskconverter becomes relevant, if the order is no
    time, range, whatever (as in mira spec)

    chaining example:
    ```var_conversion = 'z2lin,extrfromaxis2(0)'```

    Returns:
        (varconverter, maskconverter) which both are functions
    """
    if ',' in string:
        converters = [get_converter_array(s, **kwargs) for s in string.split(',')]
        varfuncs = reversed([f[0] for f in converters])
        maskfuncs = reversed([f[1] for f in converters])
        varf = lambda x: reduce(lambda r, f: f(r), varfuncs, x)
        maskf = lambda x: reduce(lambda r, f: f(r), maskfuncs, x)

        return varf, maskf


    elif string == 'since20010101':
        return lambda x: x + dt_to_ts(datetime.datetime(2001, 1, 1)), ident
    elif string == 'hours_since20150101':
        return lambda x: x*60*60 + dt_to_ts(datetime.datetime(2015, 1, 1)), ident
    elif string == 'unix':
        return lambda x: x, ident
    elif string == 'since19691231':
        return lambda x: x + dt_to_ts(datetime.datetime(1969, 12, 31, 23)), ident
    elif string == 'since19700101':
        return lambda x: x + dt_to_ts(datetime.datetime(1970, 1, 1)), ident
    elif string == 'since19040101':
        return lambda x: x + dt_to_ts(datetime.datetime(1904, 1, 1)), ident
    elif string == 'beginofday':
        if 'ncD' in kwargs.keys():
            return (lambda h: (h.astype(np.float64) * 3600. + \
                               float(dt_to_ts(datetime.datetime(kwargs['ncD'].year,
                                                                kwargs['ncD'].month,
                                                                kwargs['ncD'].day)))),
                    ident)
    elif string == "hours_since_year0":
        return (lambda x: x*24*60*60 - 62167305599.99999,
                ident)
    elif string == "pollytime":
        return (lambda x: np.array([x[i,1] + dt_to_ts(datetime.datetime.strptime(str(int(x[i,0])), "%Y%m%d"))\
                for i in range(x.shape[0])]),
                ident)
    elif string == 'since20200101':
        return lambda x: x + dt_to_ts(datetime.datetime(2020, 1, 1,)), ident


    elif string == "km2m":
        return lambda x: x * 1000., ident
    elif string == "sealevel2range":
        return lambda x: x - kwargs['altitude'], ident

    elif string == 'z2lin':
        return z2lin, ident
    elif string == 'lin2z':
        return lin2z, ident
    elif string == 'switchsign':
        return lambda x: -x, ident

    elif string == "mira_azi_offset":
        return lambda x: (x + kwargs['mira_azi_zero']) % 360, ident

    elif string == 'transposedim':
        #return np.transpose, np.transpose
        return transpose_only, transpose_only
    elif string == 'transposedim+invert3rd':
        return transpose_and_invert, transpose_and_invert
    elif string == 'divideby2':
        return divide_by(2.), ident
    elif string == 'keepNyquist':
        return ident, ident
    elif string == 'raw2Z':
        return raw2Z(**kwargs), ident
    elif string == "extract_level0":
        return lambda x: x[:, 0], ident
    elif string == "extract_level1":
        return lambda x: x[:, 1], ident
    elif string == "extract_level2":
        return lambda x: x[:, 2], ident
    elif string == 'extract_1st':
        return lambda x: np.array(x[0])[np.newaxis,], ident
    elif string == "none":
        return ident, ident
    elif 'extrfromaxis2' in string:
        return get_extrfromaxis2(string), get_extrfromaxis2(string)

    else:
        raise ValueError("converter {} not defined".format(string))


def transpose_only(var):
    return np.transpose(var)[:, :, :]


def transpose_and_invert(var):
    return np.transpose(var)[:, :, ::-1]


def divide_by(val):
    return lambda var: var / val


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
    # return (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    return datetime.datetime(2001, 1, 1) + datetime.timedelta(seconds=s)


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)


def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index  
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
            if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
                i = i + 1
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
    return 10 * np.ma.log10(array)


def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10 ** (array / 10.)


def raw2Z(array, **kwargs):
    """raw signal units (MRR-Pro) to reflectivity Z"""
    return array * kwargs['wl']**4 / (np.pi**5) / 0.93 * 10**6


def get_extrfromaxis2(string):
    """get function that extracts given index from axis2"""

    m = re.search(r"\((\d+)\)", string)
    ind = int(m.groups(0)[0])
    return lambda x: x[:,:,ind]


def fill_with(array, mask, fill):
    """fill an array where mask is true with fill value"""
    filled = array.copy()
    filled[mask] = fill
    return filled


def guess_str_to_dict(string):
    """try to convert a text string into a dict
    intended to be used in the var_def

    Returns:
        dict with flag as key and description string
    """


    if "{" in string:
        #probalby already the stringified python format
        return ast.literal_eval(string)

    elif "\nValue" in string:
        # the cloudnetpy format \nValue 0: desc\n ....
        d = {}
        for e in string.split('\nValue '):
            if len(e) > 0:
                k, v = e.split(':')
                d[int(k)] = v.strip()
        return d
    elif "\n" in string:
        # the cloudnet format 0: desc\n ....
        d = {}
        for e in string.split('\n'):
            k, v = e.split(':')
            d[int(k)] = v.strip()
        return d
    elif "\\n" in string:
        # pollynet format
        d = {}
        for e in string.split('\\n'):
            m = re.match(r'(\d{1,2}): (.*)', e.replace(r'\"', ''))
            d[int(m.group(1))] = m.group(2).strip()
        return d
    else:
        # unknown format
        return string


def _method_info_from_argv(argv=None):
    """Command-line -> method call arg processing.

    - positional args:
            a b -> method('a', 'b')
    - intifying args:
            a 123 -> method('a', 123)
    - json loading args:
            a '["pi", 3.14, null]' -> method('a', ['pi', 3.14, None])
    - keyword args:
            a foo=bar -> method('a', foo='bar')
    - using more of the above
            1234 'extras=["r2"]'  -> method(1234, extras=["r2"])

    @param argv {list} Command line arg list. Defaults to `sys.argv`.
    @returns (<method-name>, <args>, <kwargs>)

    Reference: http://code.activestate.com/recipes/577122-transform-command-line-arguments-to-args-and-kwarg/
    """
    import json
    import sys
    if argv is None:
        argv = sys.argv

    method_name, arg_strs = argv[0], argv[1:]
    args = []
    kwargs = {}
    for s in arg_strs:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            key, value = None, s
        try:
            value = json.loads(value)
        except ValueError:
            pass
        if key:
            kwargs[key] = value
        else:
            args.append(value)
    return method_name, args, kwargs


def reshape_spectra(data):
    """This function reshapes time, range and var variables of a data container and returns numpy arrays.

    Args:
        data (dict): data container

    Returns:
        list with

        - ts (numpy.array): time stamp numpy array, dim = (n_time,)
        - rg (numpy.array): range stamp numpy array, dim = (n_range,)
        - var (numpy.array): values of the spectra numpy array, dim = (n_time, n_range, n_vel)
    """
    n_ts, n_rg, n_vel = data['ts'].size, data['rg'].size, data['vel'].size

    if data['dimlabel'] == ['time', 'range', 'vel']:
        ts = data['ts'].copy()
        rg = data['rg'].copy()
        var = data['var'].copy()
        mask = data['mask'].copy()
    elif data['dimlabel'] == ['time', 'vel']:
        ts = data['ts'].copy()
        rg = np.reshape(data['rg'], (n_rg,))
        var = np.reshape(data['var'], (n_ts, 1, n_vel))
        mask = np.reshape(data['mask'], (n_ts, 1, n_vel))
    elif data['dimlabel'] == ['range', 'vel']:
        ts = np.reshape(data['ts'].copy(), (n_ts,))
        rg = data['rg'].copy()
        var = np.reshape(data['var'], (1, n_rg, n_vel))
        mask = np.reshape(data['mask'], (1, n_rg, n_vel))
    elif data['dimlabel'] == ['vel']:
        ts = np.reshape(data['ts'].copy(), (n_ts,))
        rg = np.reshape(data['rg'], (n_rg,))
        var = np.reshape(data['var'], (1, 1, n_vel))
        mask = np.reshape(data['mask'], (1, 1, n_vel))
    else:
        raise TypeError('Wrong data format in plot_spectra')

    return ts, rg, var, mask


def pformat(data, verbose=False):
    """return a pretty string from a data_container"""
    string = []
    string.append("== data container: system {} name {}  ==".format(data["system"], data["name"]))
    string.append("dimlabel    {}".format(data["dimlabel"]))
    if "time" in data["dimlabel"]:
        string.append("timestamps  {} {} to {}".format(
            data["ts"].shape,
            ts_to_dt(data["ts"][0]), ts_to_dt(data["ts"][-1])))
    elif "ts" in data.keys():
        string.append("timestamp   {}".format(ts_to_dt(data['ts'])))
    if "range" in data["dimlabel"]:
        string.append("range       {} {:7.2f} to {:7.2f}".format(
            data["rg"].shape,
            data["rg"][0], data["rg"][-1]))
        string.append("rg_unit     {}".format(data["rg_unit"]))
    elif "rg" in data.keys():
        string.append("range       {}".format(data['rg']))
        string.append("rg_unit     {}".format(data["rg_unit"]))
    if "vel" in data.keys():
        string.append("vel         {}  {:5.2f} to {:5.2f}".format(
            data["vel"].shape,
            data["vel"][0], data["vel"][-1]))
    if not np.all(data["mask"]):
        string.append("var         {}  min {:7.2e} max {:7.2e}".format(
            data['var'].shape,
            np.min(data['var'][~data['mask']]), np.max(data['var'][~data['mask']])))
        string.append("            mean {:7.2e} median {:7.2e}".format(
            np.mean(data['var'][~data['mask']]), np.median(data['var'][~data['mask']])))
    string.append("mask        {:4.1f}%".format(
        np.sum(data["mask"])/data['mask'].ravel().shape[0]*100.))
    string.append("var_unit    {}".format(data["var_unit"]))
    string.append("var_lims    {}".format(data["var_lims"]))
    string.append("default colormap {}".format(data["colormap"]))
    if verbose:
        string.append("filenames")
        string.append(pp.pformat(data["filename"], indent=2))
        string.append("paraminfo".format())
        string.append(pp.pformat(data['paraminfo'], indent=2))
    return "\n".join(string)


def isKthBitSet(n, k):
    """
    Function to check if a certain bit of a number is set (required to analyse quality flags)
    """
    if n & (1 << (k - 1)):
        return 1
    else:
        return 0


def pprint(data, verbose=False):
    """print a pretty representation of the data container"""
    print(pformat(data, verbose=verbose))


def extract_case_from_excel_sheet(data_loc, sheet_nr=0, **kwargs):
    """This function extracts information from an excel sheet. It can be used for different scenarios.
    The first row of the excel sheet contains the headline, defined as follows:

    +----+-------+-------+-------+-------+-------+-------+------------+-------+
    |    |   A   |   B   |   C   |   D   |   E   |   F   |      G     |   H   |
    +----+-------+-------+-------+-------+-------+-------+------------+-------+
    |  1 |  date | start |  end  |   h0  |  hend |  MDF  |  noise_fac | notes |
    +----+-------+-------+-------+-------+-------+-------+------------+-------+


                                OR

    +----+------------+----------+-------+-------+-----------+--------+
    |    |      A     |     B    |   C   |   D   |     E     |    F   |
    +----+------------+----------+-------+-------+-----------+--------+
    |  1 |  datestart |  dateend |   h0  |  hend | noise_fac |  notes |
    +----+------------+----------+-------+-------+-----------+--------+


    The following rows contain the cases of interest. Make sure that the ALL the data in the excel sheet is formatted as
    string! The data has to be provided in the following syntax:

        - date (string): format YYYYMMDD
        - start (string): format HHMMSS
        - datestart (string): format YYYYMMDDHHMMSS
        - dateend (string): format YYYYMMDDHHMMSS
        - end (string): format HHMMSS
        - h0 (string): minimum height
        - hend (string): maximum height
        - MDF (string): name of the MDF used for this case
        - noise_fac (string): noise factor
        - notes (string): additional notes for the case (stored but not in use by the program)

    Args:
        data_loc (string): path to the excel file (make sure the data_loc contains the suffix .xlsx)
        sheet_nr (integer): number of the desired excel sheet

    Returns:
        case_list contains the information for all cases
            
        - begin_dt (datetime object): start of the time interval
        - end_dt (datetime object): end of the time interval
        - plot_range (list): height interval
        - MDF_name (string): name of MDF used for this case
        - noisefac (string): number of standard deviations above mean noise level
        - notes (string): additional notes for the user
            
    """

    import xlrd

    excel_sheet = xlrd.open_workbook(data_loc)
    sheet = excel_sheet.sheet_by_index(sheet_nr)
    case_list = []

    if kwargs['kind'] == 'ann_input':
        # exclude header from data
        for icase in range(1, sheet.nrows):
            irow = sheet.row_values(icase)
            irow[:2] = [int(irow[i]) for i in range(2)]

            if irow[5] != 'ex':
                case_list.append({
                    'begin_dt': datetime.datetime.strptime(str(irow[0]), '%Y%m%d%H%M%S'),
                    'end_dt': datetime.datetime.strptime(str(irow[1]), '%Y%m%d%H%M%S'),
                    'plot_range': [float(irow[2]), float(irow[3])],
                    'noise_fac': irow[4],
                    'notes': irow[5]})

    if kwargs['kind'] == 'cumulustest':
        # exclude header from data
        for icase in range(1, sheet.nrows):
            irow = sheet.row_values(icase)
            irow[:3] = [int(irow[i]) for i in range(3)]

            if irow[7] != 'ex':
                case_list.append({
                    'begin_dt': datetime.datetime.strptime(str(irow[0]) + ' ' + str(irow[1]), '%Y%m%d %H%M%S'),
                    'end_dt': datetime.datetime.strptime(str(irow[0]) + ' ' + str(irow[2]), '%Y%m%d %H%M%S'),
                    'plot_range': [float(irow[3]), float(irow[4])],
                    'MDF_name': irow[5],
                    'noisefac': irow[6],
                    'notes': irow[7]})

    return case_list


def interp_only_3rd_dim(arr, old, new):
    """function to interpolate only the velocity (3rd) axis"""

    from scipy import interpolate

    f = interpolate.interp1d(old, arr, axis=2,
                             bounds_error=False, fill_value=-999.)
    new_arr = f(new)

    return new_arr


def put_in_container(data, data_container, **kwargs):
    """
    This routine will generate a new larda container, replacing var in the data_container given with data

    Kwargs:
        all keys used in larda containers, that is
        paraminfo (dict): information from params_[campaign].toml for the specific variable
        rg_unit: range unit
        mask: ndarray of same size as var, indicating masked values
        ts: time stamp in unix time
        var_lims: variable limits
        colormap: colormap to be used for plotting


    Return:
        container (dict): larda data container
    """

    container = copy.deepcopy(data_container)
    container['var'] = data
    container.update(kwargs)

    return container


def change_dir(folder_path, **kwargs):
    """
    This routine changes to a folder or creates it (including subfolders) if it does not exist already.

    Args:
        folder_path (string): path of folder to switch into
    """

    folder_path = folder_path.replace('//', '/', 1)

    if not os.path.exists(os.path.dirname(folder_path)):
        try:
            os.makedirs(os.path.dirname(folder_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    os.chdir(folder_path)
    logger.debug('\ncd to: {}'.format(folder_path))


def make_dir(folder_path):
    """
    This routine changes to a folder or creates it (including subfolders) if it does not exist already.

    Args:
        folder_path (string): path of folder to switch into
    """

    if not os.path.exists(os.path.dirname(folder_path)):
        try:
            os.makedirs(os.path.dirname(folder_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def print_traceback(txt):
    """
    Print the traceback to an error to screen.
    Args:
        - txt (string): error msg
    """
    print(txt)
    track = traceback.format_exc()
    print(track)


def smooth(y, box_pts, padding='constant'):
    """Smooth a one dimensional array using a rectangular window of box_pts points

    Args:
        y (np.array): array to be smoothed
        box_pts: number of points of the rectangular smoothing window
    Returns:
        y_smooth (np.arrax): smoothed array
    """

    box = np.ones(box_pts) / box_pts
    if padding.lower() == 'constant':
        return np.convolve(y, box, mode='full')[box_pts // 2:-box_pts // 2 + 1]
    else:
        return np.convolve(y, box, mode='same')

