
#!/usr/bin/python3

"""

"""

import numpy as np
import pyLARDA.helpers as h
from pyLARDA.NcReader import get_time_slicer, get_var_attr_from_nc
#from typing import List
import logging

logger = logging.getLogger(__name__)

def get_meta_from_bD(bD, meta_spec, varname):
    """get some meta data into the data_container

    specified within the paraminfo meta.name tags
    - gattr.name: global attribute with name
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
            meta[k] = [bD[name]]
        else:
            raise ValueError(f'meta string {v} for {k} not specified')
    return meta

def rpgfmcw_binary(paraminfo):
    """build a function for reading in time height data
    """

    def retfunc(f, time_interval, *further_intervals):
        """reading the rpg94 data with rpgypy and convert into the larda-data-format
        """
        from rpgpy import read_rpg

        logger.debug(f"filename at rpgpy binary {f}")
        header, data = read_rpg(f)

        logger.debug(f'Header: {header.keys()}')
        logger.debug(f'Data  : {data.keys()}')

        # bD binary Data (in resemblance to ncD)
        bD = {**header, **data}

        if paraminfo['ncreader'] in ['timeheight_rpg94binary']:
            try:
                range_interval = further_intervals[0]
            except:
                range_interval = []
            ranges = bD[paraminfo['range_variable']][:].astype(np.float64)
        
        times = bD[paraminfo['time_variable']][:].astype(np.float64)
        if 'time_millisec_variable' in paraminfo.keys() and \
                paraminfo['time_millisec_variable'] in bD:
            subsec = bD[paraminfo['time_millisec_variable']][:] / 1.0e3
            times += subsec
        if 'time_microsec_variable' in paraminfo.keys() and \
                paraminfo['time_microsec_variable'] in bD:
            subsec = bD[paraminfo['time_microsec_variable']][:] / 1.0e6
            times += subsec
        timeconverter, _ = h.get_converter_array(
            paraminfo['time_conversion'])
        ts = timeconverter(times)
        
        # get the time slicer from time_interval
        slicer = get_time_slicer(ts, f, time_interval)
        if slicer == None:
            return None
        
        varconverter, _ = h.get_converter_array(
            paraminfo['var_conversion'])

        if paraminfo['ncreader'] in ['timeheight_rpg94binary']:
            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])
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
        
        var = bD[paraminfo['variable_name']]
        print(f"var shape {paraminfo['variable_name']} {var.shape}")
        data = {}
        if paraminfo['ncreader'] in ['timeheight_rpg94binary']:
            data['dimlabel'] = ['time', 'range']
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
        else:
            data['dimlabel'] = ['time']
        data["filename"] = f
        data["paraminfo"] = paraminfo
        data['ts'] = ts[tuple(slicer)[0]]

        data['system'] = paraminfo['system']
        data['name'] = paraminfo['paramkey']
        data['colormap'] = paraminfo['colormap']

        if 'meta' in paraminfo:
            data['meta'] = get_meta_from_bD(bD, paraminfo['meta'], paraminfo['variable_name'])
        if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
            data['plot_varconverter'] = paraminfo['plot_varconverter']
        else:
            data['plot_varconverter'] = ''

        if paraminfo['ncreader'] in ['timeheight_rpg94binary']:
            if isinstance(times, np.ma.MaskedArray):
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]].data)
            else:
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ranges)

        data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                paraminfo, var)
        data['var_lims'] = [float(e) for e in \
                            get_var_attr_from_nc("identifier_var_lims",
                                                 paraminfo, var)]

        if "identifier_var_def" in paraminfo.keys() and not "var_def" in paraminfo.keys():
            data['var_definition'] = h.guess_str_to_dict(
                var.getncattr(paraminfo['identifier_var_def']))
        elif "var_def" in paraminfo.keys():
            data['var_definition'] =  paraminfo['var_def']

        data['var'] = varconverter(var[:])[tuple(slicer)]        

        # no getncattr available for binary data
        #if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
        #    fill_value = var.getncattr(paraminfo['identifier_fill_value'])
        #    mask = np.isclose(data['var'].data, fill_value)
        if "fill_value" in paraminfo.keys():
            fill_value = paraminfo['fill_value']
            mask = np.isclose(data['var'].data, fill_value)
        else:
            mask = ~np.isfinite(data['var'].data)

        assert not isinstance(mask, np.ma.MaskedArray), \
           "mask array shall not be np.ma.MaskedArray, but of plain booltype"
        data['mask'] = mask

        if isinstance(data['var'], np.ma.MaskedArray):
            data['var'] = data['var'].data
        assert not isinstance(data['var'], np.ma.MaskedArray), \
           "var array shall not be np.ma.MaskedArray, but of plain booltype"

        return data

    return retfunc