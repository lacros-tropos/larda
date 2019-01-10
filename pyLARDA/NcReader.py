#!/usr/bin/python3

"""

.. warning::
    reimplement this with netcdf4 ONLY

"""

import numpy as np
import netCDF4
import pyLARDA.helpers as h

import logging
logger = logging.getLogger(__name__)

def get_var_attr_from_nc(name, paraminfo, variable):
    direct_def =  name.replace("identifier_", "")
    # if both are given (eg through inheritance, choose the
    # direct definition)
    logger.debug("attr name {}".format(name))
    if name in paraminfo and direct_def not in paraminfo:
        attr = variable.getncattr(paraminfo[name])
    else:
        attr = paraminfo[name.replace("identifier_", "")]

    return attr


def reader(paraminfo):
    """build a function for reading in time height data"""
    def retfunc(f, time_interval, *further_intervals):
        """function that converts the netCDF to the larda-data-format
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:]/1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:]/1.0e6
                times += subsec

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            #print('timestamps ', ts[:5])
            # setup slice to load base on time_interval
            it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
            it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))
            it_e = it_e+1 if not it_e == ts.shape[0]-1 else None
            
            slicer = [slice(it_b, it_e)]

            if paraminfo['ncreader'] == 'timeheight' \
                    or paraminfo['ncreader'] == 'spec':
                range_tg = True

                range_interval = further_intervals[0]
                ranges = ncD.variables[paraminfo['range_variable']]
                logger.debug('loader range conversion {}'.format(paraminfo['range_conversion']))
                rangeconverter, _ = h.get_converter_array(
                    paraminfo['range_conversion'],
                    altitude=paraminfo['altitude'])
                ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e+1 if not ir_e == ranges.shape[0]-1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))

            if paraminfo['ncreader'] == 'spec':
                vel_tg = True
                slicer.append(slice(None))
            
            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'])

            var = ncD.variables[paraminfo['variable_name']]
            #print('var dict ',ncD.variables[paraminfo['variable_name']].__dict__)
            #print("time indices ", it_b, it_e)
            data = {}
            if paraminfo['ncreader'] == 'timeheight':
                data['dimlabel'] = ['time', 'range']
            elif paraminfo['ncreader'] == 'time':
                data['dimlabel'] = ['time']
            elif paraminfo['ncreader'] == 'spec':
                data['dimlabel'] = ['time', 'range', 'vel']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            
            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']
            
            if paraminfo['ncreader'] == 'timeheight' \
                    or paraminfo['ncreader'] == 'spec':
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
                data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit", 
                                                       paraminfo, ranges)
                logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))
            if paraminfo['ncreader'] == 'spec':
                if 'vel_ext_variable' in paraminfo:
                    # this special field is needed to load limrad spectra
                    # only works when vel is third variable (TODO add the dimorder shuffler)
                    vel_ext = ncD.variables[paraminfo['vel_ext_variable'][0]][int(paraminfo['vel_ext_variable'][1])]
                    vel_res = 2*vel_ext/float(var[:].shape[2])
                    data['vel'] = np.linspace(-vel_ext, +vel_ext, var[:].shape[2]) 
                    #print('vel_limrad ',data['vel'].shape, data['vel'])
                else:
                    data['vel'] = ncD.variables[paraminfo['vel_variable']][:]
            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit", 
                                                    paraminfo, var)
            data['var_lims'] = list(get_var_attr_from_nc("identifier_var_lims", 
                                                         paraminfo, var))

            # by default assume dimensions of (time, range, ...)
            # or define a custom order in the param toml file
            if 'dimorder' in paraminfo:
                slicer = [slicer[i] for i in paraminfo['dimorder']]

            if "identifier_fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                mask = (var[tuple(slicer)].data == fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo['fill_value']
                mask = np.isclose(var[tuple(slicer)].data, fill_value)
            else:
                mask = ~np.isfinite(var[tuple(slicer)].data)

            data['var'] = varconverter(var[tuple(slicer)].data)
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
        flvl0 = f.replace("LV1", "LV0")
        with netCDF4.Dataset(flvl0) as ncD:
            ch1range = ncD.variables['C1Range']
            ch2range = ncD.variables['C2Range']
            ch3range = ncD.variables['C3Range']

            ranges = np.hstack([ch1range[:], ch2range[:], ch3range[:]])

        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)
            if 'time_millisec_variable' in paraminfo.keys() and \
                    paraminfo['time_millisec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_millisec_variable']][:]/1.0e3
                times += subsec
            if 'time_microsec_variable' in paraminfo.keys() and \
                    paraminfo['time_microsec_variable'] in ncD.variables:
                subsec = ncD.variables[paraminfo['time_microsec_variable']][:]/1.0e6
                times += subsec
            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            #print('timestamps ', ts[:5])
            # setup slice to load base on time_interval
            it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
            it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))
            it_e = it_e+1 if not it_e == ts.shape[0]-1 else None

            slicer = [slice(it_b, it_e)]
            
            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])

            varconverter, _ = h.get_converter_array(
                paraminfo['var_conversion'])
            
            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if not range_interval[1] == 'max':
                ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                ir_e = ir_e+1 if not ir_e == ranges.shape[0]-1 else None
            else:
                ir_e = None

            slicer.append(slice(ir_b, ir_e))

            ch1var = ncD.variables['C1'+paraminfo['variable_name']]
            ch2var = ncD.variables['C2'+paraminfo['variable_name']]
            ch3var = ncD.variables['C3'+paraminfo['variable_name']]
            #print('var dict ',ch1var.__dict__)
            #print('shapes ', ts.shape, ch1range.shape, ch1var.shape)
            #print("time indices ", it_b, it_e)
            data = {}
            data['dimlabel'] = ['time', 'range']
            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            
            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']
            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit", 
                                                   paraminfo, ch1range)
            data['var_unit'] = get_var_attr_from_nc("identifier_var_unit", 
                                                    paraminfo, ch1var)
            data['var_lims'] = list(get_var_attr_from_nc("identifier_var_lims", 
                                                         paraminfo, ch1var))
            
            var = np.hstack([ch1var[:], ch2var[:], ch3var[:]])

            if "identifier_fill_value" in paraminfo.keys():
                fill_value = var.getncattr(paraminfo['identifier_fill_value'])
                data['mask'] = (var[tuple(slicer)].data==fill_value)
            elif "fill_value" in paraminfo.keys():
                fill_value = paraminfo["fill_value"]
                data['mask'] = np.isclose(var[tuple(slicer)], fill_value)
            else:
                data['mask'] = ~np.isfinite(var[tuple(slicer)].data)
            data['var'] = varconverter(var[tuple(slicer)].data)

            return data

    return retfunc

