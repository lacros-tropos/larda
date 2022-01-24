#!/usr/bin/python3

"""

Try to use netcdf4-python

"""

import numpy as np
import netCDF4
import pyLARDA.helpers as h
from typing import List
import logging
import datetime
import scipy

logger = logging.getLogger(__name__)


def get_time_slicer(
    ts: np.array, 
    f: str, 
    time_interval: List[datetime.datetime]) -> list:
    """get time slicer from the time_interval
    Following options are available

    1. time_interval with [ts_begin, ts_end]
    2. only one timestamp is selected and the found right one would be beyond the ts range -> argnearest instead searchsorted
    3. only one is timestamp

    Args:
        ts: unix timestamps as array
        f: filename
        time_interval: time interval to slice for

    Returns:
        slicer


    """

    # select first timestamp right of begin (not left if nearer as above)
    #print(f'start time {h.ts_to_dt(ts[0])}')
    it_b = 0 if ts.shape[0] == 1 else np.searchsorted(ts, h.dt_to_ts(time_interval[0]), side='right')
    if len(time_interval) == 2:
        it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))

        if it_b == ts.shape[0]: it_b = it_b - 1
        valid_step =  3 * np.median(np.diff(ts))
        if ts[it_e] < h.dt_to_ts(time_interval[0]) - valid_step or ts[it_b] < h.dt_to_ts(time_interval[0]):
            # second condition is to ensure that no timestamp before
            # the selected interval is choosen
            # (problem with limrad after change of sampling frequency)
            str = 'found last profile of file {}\n at ts[it_e] {} too far ({}s) from {}\n'.format(
                    f, h.ts_to_dt(ts[it_e]), valid_step, time_interval[0]) \
                 + 'or begin too early {} < {}\n returning None'.format(h.ts_to_dt(ts[it_b]), time_interval[0])
            logger.warning(str)
            return None

        it_e = it_e + 1 if not it_e == ts.shape[0] - 1 else None
        slicer = [slice(it_b, it_e)]
    elif it_b == ts.shape[0]:
        # only one timestamp is selected
        # and the found right one would be beyond the ts range
        it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
        slicer = [slice(it_b, it_b + 1)]
    else:
        slicer = [slice(it_b, it_b + 1)]
    return slicer


def get_var_attr_from_nc(name, paraminfo, variable):
    """get the attribute from the variable

    Args:
        name:
        paraminfo:
        variable:
    """

    direct_def = name.replace("identifier_", "")
    # if both are given (eg through inheritance, choose the
    # direct definition)
    logger.debug("attr name {}".format(name))
    attr = ''
    if name in paraminfo and direct_def not in paraminfo:
        try:
            attr = variable.getncattr(paraminfo[name])
        except Exception as e:
            logger.warning('Error extracting paraminfo of variable ' + str(name))
            logger.warning('Check spelling in .toml file or remove from .toml')
            logger.warning('Exception :: ', e)
    else:
        attr = paraminfo[name.replace("identifier_", "")]

    return attr


def get_meta_from_nc(ncD, meta_spec, varname):
    """get some meta data into the data_container

    specified within the paraminfo meta.name tags
    - gattr.name: global attribute with name
    - vattr.name: variable attribute with name
    - var.name: additional varable (ideally single value)

    Args:
        ncD: netCDF file object
        meta_spec: dict with all meta. definition
        varname: name of the variable to load (for vattr)

    Returns:
        dict with meta data
    """

    meta = {}
    for k, v in meta_spec.items():
        where, name = v.split('.')
        if where == 'gattr':
            meta[k] = [ncD.getncattr(name)]
        elif where == 'vattr':
            meta[k] = [ncD.variables[varname].getncattr(name)]
        elif where == 'var':
            meta[k] = [ncD.variables[name][:].data.tolist()]
        else:
            raise ValueError(f'meta string {v} for {k} not specified')
    return meta


def reader(paraminfo):
    """build a function for reading in time height data"""

    def retfunc(f, time_interval, *further_intervals):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            if 'auto_mask_scale' in paraminfo and paraminfo['auto_mask_scale'] == False:
                ncD.set_auto_mask(False)

            varconv_args = {}
            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec
            if 'base_time_variable' in paraminfo.keys() and \
                    paraminfo['base_time_variable'] in ncD.variables:
                basetime = ncD.variables[paraminfo['base_time_variable']][:].astype(np.float64)
                times += basetime

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            if isinstance(times, np.ma.MaskedArray):
                ts = timeconverter(times.data)
            else:
                ts = timeconverter(times)
            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer is None and paraminfo['ncreader'] != 'pollynet_profile':
                logger.critical(f'No time slice found!\nfile :: {f}\n')
                return None

            if paraminfo['ncreader'] == "pollynet_profile":
                slicer = [slice(None)]

            if paraminfo['ncreader'] in ['timeheight', 'spec', 'mira_noise', 'pollynet_profile']:
                range_tg = True

                try:
                    range_interval = further_intervals[0]
                except IndexError as e:
                    logger.error('No range interval was given.')

                ranges = ncD.variables[paraminfo['range_variable']]
                logger.debug('loader range conversion {}'.format(paraminfo['range_conversion']))
                rangeconverter, _ = h.get_converter_array(
                    paraminfo['range_conversion'],
                    altitude=paraminfo['altitude'])
                ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
                if len(range_interval) == 2:
                    if not range_interval[1] == 'max':
                        ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                        ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                    else:
                        ir_e = None
                    slicer.append(slice(ir_b, ir_e))
                else:
                    slicer.append(slice(ir_b, ir_b + 1))

            if paraminfo['ncreader'] == 'spec':
                if 'compute_velbins' in paraminfo:
                    if paraminfo['compute_velbins'] == "mrrpro":
                        wl = 1.238*10**(-2) # wavelength (fixed) - 24 GHz
                        varconv_args.update({"wl": wl})
                vel_tg = True
                slicer.append(slice(None))
            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'], **varconv_args)
            if 'vel_conversion' in paraminfo:
                velconverter, _ = h.get_converter_array(paraminfo['vel_conversion'])

            var = ncD.variables[paraminfo['variable_name']]
            # print('var dict ',ncD.variables[paraminfo['variable_name']].__dict__)
            # print("time indices ", it_b, it_e)
            data = {}
            if paraminfo['ncreader'] == 'timeheight':
                data['dimlabel'] = ['time', 'range']
            elif paraminfo['ncreader'] == 'time':
                data['dimlabel'] = ['time']
            elif paraminfo['ncreader'] == 'spec':
                data['dimlabel'] = ['time', 'range', 'vel']
            elif paraminfo['ncreader'] == 'mira_noise':
                data['dimlabel'] = ['time', 'range']
            elif paraminfo['ncreader'] == "pollynet_profile":
                data['dimlabel'] = ['time', 'range']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            if paraminfo['ncreader'] in ['timeheight', 'spec', 'mira_noise', 'pollynet_profile']:
                if isinstance(times, np.ma.MaskedArray):
                    data['rg'] = rangeconverter(ranges[tuple(slicer)[1]].data)
                else:
                    data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])

                data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                       paraminfo, ranges)
                logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))
            if paraminfo['ncreader'] == 'spec':
                if 'vel_ext_variable' in paraminfo:
                    # this special field is needed to load limrad spectra
                    vel_ext = ncD.variables[paraminfo['vel_ext_variable'][0]][int(paraminfo['vel_ext_variable'][1])]
                    vel_res = 2 * vel_ext / float(var[:].shape[2])
                    data['vel'] = np.linspace(-vel_ext + (0.5 * vel_res),
                                              +vel_ext - (0.5 * vel_res),
                                              var[:].shape[2])
                elif 'compute_velbins' in paraminfo:
                    if paraminfo['compute_velbins'] == 'mrrpro':
                    # this is used to read in MRR-PRO spectra
                        fs = 500000 # sampling rate of MRR-Pro (fixed)
                        vel_ext = fs/4/ncD.dimensions['range'].size*wl
                        vel_res = vel_ext / float(var[:].shape[2])
                        data['vel'] = np.linspace(0 - (0.5 * vel_res),
                                              -vel_ext + (0.5 * vel_res),
                                              var[:].shape[2])
                else:
                    data['vel'] = ncD.variables[paraminfo['vel_variable']][:]
                if 'vel_conversion' in paraminfo:
                    data['vel'] = velconverter(data['vel'])

            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]

            # by default assume dimensions of (time, range, ...)
            # or define a custom order in the param toml file
            if 'dimorder' in paraminfo:
                slicer = [slicer[i] for i in paraminfo['dimorder']]

            if paraminfo['ncreader'] == "pollynet_profile":
                del slicer[0]

            # read in the variable definition dictionary
            #
            if "identifier_var_def" in paraminfo.keys() and not "var_def" in paraminfo.keys():
                data['var_definition'] = h.guess_str_to_dict(
                    var.getncattr(paraminfo['identifier_var_def']))
            elif "var_def" in paraminfo.keys():
                data['var_definition'] =  paraminfo['var_def']

            if paraminfo['ncreader'] == 'mira_noise':
                r_c = ncD.variables[paraminfo['radar_const']][:]
                snr_c = ncD.variables[paraminfo['SNR_corr']][:]
                npw = ncD.variables[paraminfo['noise_pow']][:]
                calibrated_noise = r_c[slicer[0], np.newaxis] * var[tuple(slicer)].data * snr_c[tuple(slicer)].data / \
                                   npw[slicer[0], np.newaxis] * (data['rg'][np.newaxis, :] / 5000.) ** 2
                data['var'] = calibrated_noise
            else:
                data['var'] = varconverter(var[:])[tuple(slicer)]

                #if paraminfo['compute_velbins'] == "mrrpro":
                #    data['var'] = data['var'] * wl** 4 / (np.pi** 5) / 0.93 * 10**6

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                mask = np.isclose(data['var'].data, fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo['fill_value']
                mask = np.isclose(data['var'].data, fill_value)
            else:
                mask = ~np.isfinite(data['var'].data)
            
            #if isinstance(mask, np.ma.MaskedArray):
            #    mask = mask.mask
            assert not isinstance(mask, np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"
            data['mask'] = np.logical_or(mask, data['var'].mask)

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            if paraminfo['ncreader'] == "pollynet_profile":
                data['var'] = data['var'][np.newaxis, :]
                data['mask'] = data['mask'][np.newaxis, :]

            return data

    return retfunc


def auxreader(paraminfo):
    """build a function for reading in time height data"""

    def retfunc(f, time_interval, *further_intervals):
        """function that converts the netCDF to the larda-data-format
        this one is for aux values that don't have a dedicated time domain
        (nevertheless the time is read in, to estimate the coverage of the file)
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times.data)

            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer == None and paraminfo['ncreader'] != 'aux_all_ts':
                return None

            if paraminfo['ncreader'] == "aux_all_ts":
                slicer = [slice(None)]

            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'])

            var = ncD.variables[paraminfo['variable_name']]
            # print('var dict ',ncD.variables[paraminfo['variable_name']].__dict__)
            # print("time indices ", it_b, it_e)
            data = {}
            data['dimlabel'] = ['time', 'aux']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[0:1]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                mask = (var[:] == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo['fill_value']
                mask = np.isclose(var[:], fill_value)
            else:
                mask = ~np.isfinite(var[:])

            if isinstance(mask, np.ma.MaskedArray):
                mask = mask.data
            assert not isinstance(mask, np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            data['var'] = varconverter(var[:])

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            data['mask'] = maskconverter(mask)

            return data

    return retfunc


def timeheightreader_rpgfmcw(paraminfo):
    """build a function for reading in time height data
    special function for a special instrument ;)

    the issues are:

    - range variable in different file
    - stacking of single variables

    for now works only with 3 chirps and range variable only in level0
    """

    def retfunc(f, time_interval, range_interval):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            no_chirps = ncD.dimensions['Chirp'].size

            ranges_per_chirp = [
                ncD.variables['C{}Range'.format(i + 1)] for i in range(no_chirps)]
            ch1range = ranges_per_chirp[0]

            ranges = np.hstack([rg[:] for rg in ranges_per_chirp])


            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec
            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer == None:
                return None

            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])

            varconverter, _ = h.get_converter_array(
                paraminfo['var_conversion'])

            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))

            no_chirps = ncD.dimensions['Chirp'].size

            var_per_chirp = [
                ncD.variables['C{}'.format(i + 1) + paraminfo['variable_name']] for i in range(no_chirps)]
            ch1var = var_per_chirp[0]

            # ch1var = ncD.variables['C1'+paraminfo['variable_name']]
            # ch2var = ncD.variables['C2'+paraminfo['variable_name']]
            # ch3var = ncD.variables['C3'+paraminfo['variable_name']]

            # print('var dict ',ch1var.__dict__)
            # print('shapes ', ts.shape, ch1range.shape, ch1var.shape)
            # print("time indices ", it_b, it_e)
            data = {}
            data['dimlabel'] = ['time', 'range']
            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ch1range)
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, ch1var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, ch1var)]
            var = np.hstack([v[:] for v in var_per_chirp])
            # var = np.hstack([ch1var[:], ch2var[:], ch3var[:]])

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                data['mask'] = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo["fill_value"]
                data['mask'] = np.isclose(var[tuple(slicer)].data, fill_value)
            else:
                data['mask'] = ~np.isfinite(var[tuple(slicer)].data)

            assert not isinstance(data['mask'], np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"
            data['var'] = varconverter(var[tuple(slicer)].data)

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            return data

    return retfunc


def specreader_rpgfmcw(paraminfo):
    """build a function for reading in spectral data
    special function for a special instrument ;)

    the issues are:

    - range variable in different file
    - stacking of single variables

    for now works only with 3 chirps and range variable only in level0
    """

    def retfunc(f, time_interval, range_interval):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))

        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec
            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            no_chirps = ncD.dimensions['Chirp'].size

            ranges_per_chirp = [
                ncD.variables['C{}Range'.format(i + 1)] for i in range(no_chirps)]
            ch1range = ranges_per_chirp[0]

            ranges = np.hstack([rg[:] for rg in ranges_per_chirp])

            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer == None:
                return None

            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])

            varconverter, _ = h.get_converter_array(
                paraminfo['var_conversion'])

            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))

            vars_per_chirp = [
                ncD.variables['C{}{}'.format(i + 1, paraminfo['variable_name'])] for i in range(no_chirps)]
            ch1var = vars_per_chirp[0]
            # print('var dict ',ch1var.__dict__)
            # print('shapes ', ts.shape, ch1range.shape, ch1var.shape)
            # print("time indices ", it_b, it_e)

            data = {}
            data['dimlabel'] = ['time', 'range', 'vel']
            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ch1range)
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, ch1var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, ch1var)]
            if 'vel_ext_variable' in paraminfo:
                # define the function
                get_vel_ext = lambda i: ncD.variables[paraminfo['vel_ext_variable'][0]][:][i]
                # apply it to every chirp
                vel_ext_per_chirp = [get_vel_ext(i) for i in range(no_chirps)]

                vel_dim_per_chirp = [v.shape[2] for v in vars_per_chirp]
                calc_vel_res = lambda v_e, v_dim: 2.0 * v_e / float(v_dim)
                vel_res_per_chirp = [calc_vel_res(v_e, v_dim) for v_e, v_dim \
                                     in zip(vel_ext_per_chirp, vel_dim_per_chirp)]

                # for some very obscure reason lambda is not able to unpack 3 values
                def calc_vel(vel_ext, vel_res, v_dim):
                    return np.linspace(-vel_ext + (0.5 * vel_res),
                                       +vel_ext - (0.5 * vel_res),
                                       v_dim)

                vel_per_chirp = [calc_vel(v_e, v_res, v_dim) for v_e, v_res, v_dim \
                                 in zip(vel_ext_per_chirp, vel_res_per_chirp, vel_dim_per_chirp)]
            else:
                raise NotImplemented("other means of getting the var dimension are not implemented yet")
            data['vel'] = vel_per_chirp[0]

            # interpolate the variables here
            if 'var_conversion' in paraminfo and paraminfo['var_conversion'] == 'keepNyquist':
                # the interpolation is only done for the number of spectral lines, not the velocity itself
                quot = [i/vel_dim_per_chirp[0] for i in vel_dim_per_chirp[1:]]
                vars_interp = [vars_per_chirp[0]]
                ich = 1
                for var, vel in zip(vars_per_chirp[1:], vel_per_chirp[1:]):
                    data['vel_ch{}'.format(ich+1)] = vel_per_chirp[ich]
                    new_vel = np.linspace(vel[0], vel[-1], vel_dim_per_chirp[0])
                    vars_interp.append(interp_only_3rd_dim(var[:] * quot[ich-1], vel, new_vel, kind='nearest'))
                    ich += 1
            else:
                vars_interp = [vars_per_chirp[0]] + \
                              [interp_only_3rd_dim(var, vel, vel_per_chirp[0]) \
                               for var, vel in zip(vars_per_chirp[1:], vel_per_chirp[1:])]


            var = np.hstack([v[:] for v in vars_interp])
            logger.debug('interpolated spectra from\n{}\n{} to\n{}'.format(
                [v[:].shape for v in vars_per_chirp],
                ['{:5.3f}'.format(vel[0]) for vel in vel_per_chirp],
                [v[:].shape for v in vars_interp]))
            logger.info('var.shape interpolated spectra {}'.format(var.shape))

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                data['mask'] = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo["fill_value"]
                data['mask'] = np.isclose(var[tuple(slicer)].data, fill_value)
            else:
                data['mask'] = ~np.isfinite(var[tuple(slicer)].data)
            if isinstance(times, np.ma.MaskedArray):
                data['var'] = varconverter(var[tuple(slicer)].data)
            else:
                data['var'] = varconverter(var[tuple(slicer)].data)


            assert not isinstance(data['mask'], np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            return data

    return retfunc


def specreader_rpgpy(paraminfo):
    """build a function for reading in spectral data
    """

    def retfunc(f, time_interval, range_interval):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))

        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec
            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            no_chirps = ncD.dimensions['chirp'].size

            # check if spectra will be interpolated over chirps or if we extract only one chirp
            interpolate_velocity = paraminfo['paramkey'][0] != 'C'
            if not interpolate_velocity:
                print(f'Key {paraminfo["paramkey"]} starts with "C", only reading chirp {paraminfo["paramkey"][1]}. ')
            ranges = ncD.variables[paraminfo['range_variable']]

            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer == None:
                return None

            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])

            varconverter, _ = h.get_converter_array(
                paraminfo['var_conversion'])

            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))

            var = ncD.variables[paraminfo['variable_name']]

            data = {}
            data['dimlabel'] = ['time', 'range', 'vel']
            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ranges)
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]
            if no_chirps > 1:
                range_offsets = np.hstack((ncD.variables['chirp_start_indices'][:], ncD.variables['n_range_layers'][:]))
                if interpolate_velocity:
                    common_velocity = np.linspace(-np.nanmax(ncD.variables['velocity_vectors']),
                                                  np.nanmax(ncD.variables['velocity_vectors']),
                                                  ncD.variables['velocity_vectors'][0].shape[0])
                if interpolate_velocity:
                    var_array = []
                    for i in range(no_chirps):
                        v = ncD.variables['velocity_vectors'][:][i]
                        valid_indices = v != -999
                        dv = np.nanmean(np.diff(v[valid_indices]))
                        var_per_chirp = var[tuple(slicer)[0], range_offsets[i]: range_offsets[i+1], valid_indices] / dv
                        f = scipy.interpolate.interp1d(v[valid_indices], var_per_chirp, bounds_error=False, fill_value=0.)
                        v_interp = f(common_velocity)
                        var_array.append(v_interp)

                    var_array = np.hstack(var_array)
                    dv2 = common_velocity[5] - common_velocity[4]
                    var_out = var_array*dv2

                    data['vel'] = common_velocity
                else:
                    chirp_to_extract = int(paraminfo['paramkey'][1])
                    assert(chirp_to_extract <= no_chirps), f"chirp to extract is {chirp_to_extract} but number of chirps" \
                                                           f" is {no_chirps}."
                    var_out = np.zeros(var.shape)
                    v = ncD.variables['velocity_vectors'][:][chirp_to_extract-1]
                    valid_indices = ~v.mask
                    var_out[tuple(slicer)[0], range_offsets[chirp_to_extract-1]: range_offsets[chirp_to_extract], :] = \
                        var[tuple(slicer)[0], range_offsets[chirp_to_extract-1]: range_offsets[chirp_to_extract], :]

                    var_out = var_out[tuple(slicer)[0], :, valid_indices]
                    data['vel'] = v[valid_indices]

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                data['mask'] = (var_out[:, tuple(slicer)[1]] == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo["fill_value"]
                data['mask'] = np.isclose(var_out[:, tuple(slicer)[1]], fill_value)
            else:
                data['mask'] = ~np.isfinite(var_out[:, tuple(slicer)[1]])
            if isinstance(times, np.ma.MaskedArray):
                data['var'] = varconverter(var_out[:, tuple(slicer)[1]])
            else:
                data['var'] = varconverter(var_out[:, tuple(slicer)[1]])

            assert not isinstance(data['mask'], np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            return data

    return retfunc


def scanreader_mira(paraminfo):
    """reader for the scan files

    - load full file regardless of selected time
    - covers spec_timeheight and spec_time
    """

    def retfunc(f, time_interval, *further_intervals):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            if isinstance(times, np.ma.MaskedArray):
                ts = timeconverter(times.data)
            else:
                ts = timeconverter(times)

            # load the whole time-range from the file
            slicer = [slice(None)]

            if paraminfo['ncreader'] == 'scan_timeheight':
                range_tg = True

                range_interval = further_intervals[0]
                ranges = ncD.variables[paraminfo['range_variable']]
                logger.debug('loader range conversion {}'.format(paraminfo['range_conversion']))
                rangeconverter, _ = h.get_converter_array(
                    paraminfo['range_conversion'],
                    altitude=paraminfo['altitude'])
                ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
                if len(range_interval) == 2:
                    if not range_interval[1] == 'max':
                        ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                        ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                    else:
                        ir_e = None
                    slicer.append(slice(ir_b, ir_e))
                else:
                    slicer.append(slice(ir_b, ir_b + 1))

            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'],
                mira_azi_zero=paraminfo['mira_azi_zero'])

            var = ncD.variables[paraminfo['variable_name']]
            # print('var dict ',ncD.variables[paraminfo['variable_name']].__dict__)
            # print("time indices ", it_b, it_e)
            data = {}
            if paraminfo['ncreader'] == 'scan_timeheight':
                data['dimlabel'] = ['time', 'range']
            elif paraminfo['ncreader'] == 'scan_time':
                data['dimlabel'] = ['time']
            # elif paraminfo['ncreader'] == 'spec':
            #    data['dimlabel'] = ['time', 'range', 'vel']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            if paraminfo['ncreader'] == 'scan_timeheight':
                if isinstance(times, np.ma.MaskedArray):
                    data['rg'] = rangeconverter(ranges[tuple(slicer)[1]].data)
                else:
                    data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])

                data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                       paraminfo, ranges)
                logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))
            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]

            # by default assume dimensions of (time, range, ...)
            # or define a custom order in the param toml file
            if 'dimorder' in paraminfo:
                slicer = [slicer[i] for i in paraminfo['dimorder']]

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                mask = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo['fill_value']
                mask = np.isclose(var[tuple(slicer)].data, fill_value)
            else:
                mask = ~np.isfinite(var[tuple(slicer)].data)

            assert not isinstance(mask, np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            data['var'] = varconverter(var[tuple(slicer)].data)

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            data['mask'] = maskconverter(mask)

            return data

    return retfunc


def interp_only_3rd_dim(arr, old, new, **kwargs):
    """function to interpolate only the velocity (3rd) axis"""

    from scipy import interpolate

    kind_ = kwargs['kind'] if 'kind' in kwargs else 'linear'

    f = interpolate.interp1d(old, arr, axis=2,
                             bounds_error=False, fill_value=-999., kind=kind_)
    new_arr = f(new)

    return new_arr


def specreader_kazr(paraminfo):
    """build a function for reading in spectral data
    another special function for another special instrument ;)

    the issues are:

    - variables time and range are merged and can be accessed by a locator mask
    - noise is not saved and has to be computed from the spectra
    - spectra are not in reflectivity but in 10*log10(mW)
    - need to be converted using specZ = spec * h.z2lin(float(f.cal_constant[:-3])) * self.range[ir]**2

    """

    def retfunc(f, time_interval, range_interval):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))

        with netCDF4.Dataset(f, 'r') as ncD:
            ranges = ncD.variables[paraminfo['range_variable']]
            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            locator_mask = ncD.variables[paraminfo['mask_var']][:].astype(np.int)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:] / 1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:] / 1.0e6
                times += subsec
            if 'base_time_variable' in paraminfo.keys() and \
                    paraminfo['base_time_variable'] in ncD.variables:
                basetime = ncD.variables[paraminfo['base_time_variable']][:].astype(np.float64)
                times += basetime
            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            it_b = np.searchsorted(ts, h.dt_to_ts(time_interval[0]), side='right')
            if len(time_interval) == 2:
                it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))
                if it_b == ts.shape[0]: it_b = it_b - 1
                if ts[it_e] < h.dt_to_ts(time_interval[0]) - 3 * np.median(np.diff(ts)) \
                        or ts[it_b] < h.dt_to_ts(time_interval[0]):
                    # second condition is to ensure that no timestamp before
                    # the selected interval is chosen
                    # (problem with limrad after change of sampling frequency)
                    logger.warning(
                        'last profile of file {}\n at {} too far from {}'.format(
                            f, h.ts_to_dt(ts[it_e]), time_interval[0]))
                    return None
                it_e = it_e + 1 if not it_e == ts.shape[0] - 1 else None
                slicer = [slice(it_b, it_e)]
            elif it_b == ts.shape[0]:
                # only one timestamp is selected
                # and the found right one would be beyond the ts range
                it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
                slicer = [slice(it_b, it_b + 1)]
            else:
                slicer = [slice(it_b, it_b + 1)]

            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])

            varconverter, _ = h.get_converter_array(
                paraminfo['var_conversion'])

            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))

            range_out = rangeconverter(ranges[tuple(slicer)[1]])
            cal = getattr(ncD, paraminfo['cal_const'])
            var = ncD.variables[paraminfo['variable_name']][:].astype(np.float64)
            var = var[locator_mask]
            vel = ncD.variables[paraminfo['vel_variable']][:].astype(np.float64)
            # print('var dict ',ch1var.__dict__)
            # print('shapes ', ts.shape, ch1range.shape, ch1var.shape)
            # print("time indices ", it_b, it_e)

            data = {}
            data['dimlabel'] = ['time', 'range', 'vel']
            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            data['rg'] = range_out

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ranges)
            #data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
            #                                        paraminfo, var)
            data['var_unit'] = 'dBZ m-1 s'
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]
            data['vel'] = vel

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                data['mask'] = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo["fill_value"]
                data['mask'] = np.isclose(var[tuple(slicer)], fill_value)
            elif "mask_var" in paraminfo.keys():
                # combine locator mask and mask of infinite values
                mask = locator_mask.mask[tuple(slicer)]
                data["mask"] = np.logical_or(~np.isfinite(var[tuple(slicer)].data), np.repeat(mask[:,:,np.newaxis],len(data['vel']), axis=2))
            else:
                data['mask'] = ~np.isfinite(var[tuple(slicer)].data)
            if isinstance(times, np.ma.MaskedArray):
                var = varconverter(var[tuple(slicer)].data)
            else:
                var = varconverter(var[tuple(slicer)])

            assert not isinstance(mask, np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            var2 = h.z2lin(var) * h.z2lin(float(cal[:-3])) * (range_out ** 2)[np.newaxis, :, np.newaxis]
            data['var'] = var2

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            return data

    return retfunc


def reader_pollyraw(paraminfo):
    """build a function for reading in the polly raw data into larda"""

    def retfunc(f, time_interval, *further_intervals):
        """function that converts the netCDF to the larda container
        """
        logger.debug("filename at reader {}".format(f))
        import zipfile
        import os
        with zipfile.ZipFile(f) as zfile:
            path, file = os.path.split(f)
            ncD = netCDF4.Dataset('dummy', mode='r',
                                  memory=zfile.read(file[:-4]))

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            if isinstance(times, np.ma.MaskedArray):
                ts = timeconverter(times.data)
            else:
                ts = timeconverter(times)
            # get the time slicer from time_interval
            slicer = get_time_slicer(ts, f, time_interval)
            if slicer == None:
                return None

            # load just the first 2500 range bins of polly
            slicer.append(slice(0, 2500))
            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'])

            varname, dim = paraminfo['variable_name'].split(':')
            slicer.append(int(dim))
            var = ncD.variables[varname]
            # print('var dict ',ncD.variables[paraminfo['variable_name']].__dict__)
            # print("time indices ", it_b, it_e)
            data = {}
            data['dimlabel'] = ['time', 'range']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            # also experimental: vis_varconverter
            if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
                data['plot_varconverter'] = paraminfo['plot_varconverter']
            else:
                data['plot_varconverter'] = ''

            data['rg'] = np.arange(0, 2500)

            data['rg_unit'] = 'range_bin'
            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                get_var_attr_from_nc("identifier_var_lims",
                                                     paraminfo, var)]

            # by default assume dimensions of (time, range, ...)
            # or define a custom order in the param toml file
            if 'dimorder' in paraminfo:
                slicer = [slicer[i] for i in paraminfo['dimorder']]

            if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                mask = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo['fill_value']
                mask = np.isclose(var[tuple(slicer)].data, fill_value)
            else:
                mask = ~np.isfinite(var[tuple(slicer)].data)

            assert not isinstance(mask, np.ma.MaskedArray), \
               "mask array shall not be np.ma.MaskedArray, but of plain booltype"

            #print(slicer)
            data['var'] = varconverter(var[tuple(slicer)].data)

            if isinstance(data['var'], np.ma.MaskedArray):
                data['var'] = data['var'].data
            assert not isinstance(data['var'], np.ma.MaskedArray), \
               "var array shall not be np.ma.MaskedArray, but of plain booltype"

            data['mask'] = maskconverter(mask)

            return data

    return retfunc


def reader_wyoming_sounding(paraminfo):
    """
    build a reader to read in Wyoming Upper Air soundings, saved locally as a txt file
    Args:
        paraminfo: parameter information from toml file

    Returns:
        reader function

    """
    def retfunc(f, time_interval, *further_intervals):
        """
        function that converts the txt file to larda data container
        Args:
            f:
            time_interval:

        Returns:
            larda data container with sounding data
        """
        import csv
        import datetime
        logger.debug("filename at reader {}".format(f))
        with open(f) as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader, None)
            var_index = [i for i,j in enumerate(headers) if j == paraminfo['variable_name']]
            assert(len(var_index) == 1), "mismatch between headers in file and variable name in toml"
            rg_index = [i for i,j in enumerate(headers) if j == paraminfo['range_variable']]
            assert(len(rg_index) == 1), "mismatch between headers in file and range variable name in toml"
            data = {}
            data['dimlabel'] = ['time', 'range']
            data['ts'] = np.array([h.dt_to_ts(datetime.datetime.strptime(f.name.split('/')[-1][0:11], '%Y%m%d_%H'))])
            data['var'] = []
            data['rg'] = []
            for row in reader:
                try:
                    data['var'].append(float(row[var_index[0]]))
                except ValueError:  # empty line cannot be converted to float
                    data['var'].append(np.nan)
                data['rg'].append(float(row[rg_index[0]]))
            data['var'] = np.array(data['var'])[np.newaxis,:]
            data['rg'] = np.array(data['rg'])
            data['mask'] = np.isnan(data['var'])
            data['name'] = paraminfo['paramkey']
            data['system'] = paraminfo['system']
            data['var_lims'] = paraminfo['var_lims']
            data['colormap'] = 'jet' if not 'colormap' in paraminfo else paraminfo['colormap']
            data['rg_unit'] = paraminfo['rg_unit']
            data['var_unit'] = paraminfo['var_unit']
            data['paraminfo'] = paraminfo
            data['filename'] = f.name
            return data

    return retfunc
