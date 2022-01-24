#!/usr/bin/python3


import datetime
import sys

import matplotlib
import numpy as np
from copy import copy

# import itertools

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
# scientific python imports
import scipy.interpolate
from scipy import stats
import xarray as xr
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional

import pyLARDA.VIS_Colormaps as VIS_Colormaps
import pyLARDA.helpers as h

import logging

logger = logging.getLogger(__name__)


def join(datadict1, datadict2):
    """join two data containers in time domain
    
    when you want to join more than two data containers use reduce as done
    in the connector

    they have to be in the correct time order

    Args:
        datadict1: left data container
        datadict2: right data container

    Returns:
        merged data container
    """
    new_data = {}
    assert datadict1['dimlabel'] == datadict2['dimlabel'], \
        f"{datadict1['dimlabel']} and {datadict2['dimlabel']} do not match"
    new_data['dimlabel'] = datadict1['dimlabel']
    container_type = datadict1['dimlabel']

    if container_type == ['time', 'range']:
        logger.debug("{} {} {}".format(
            datadict1['ts'].shape, datadict1['rg'].shape, datadict1['var'].shape))
        logger.debug("{} {} {}".format(
            datadict2['ts'].shape, datadict2['rg'].shape, datadict2['var'].shape))
    thisjoint = datadict1['ts'].shape[0]
    new_data["joints"] = datadict1.get('joints', []) + [thisjoint] + datadict2.get('joints', [])
    logger.debug("joints {}".format(new_data['joints']))
    new_data['filename'] = h.flatten([datadict1['filename']] + [datadict2['filename']])
    if 'meta' in datadict1:
        new_data['meta'] = {}
        for k in datadict1['meta']:
            new_data['meta'][k] = h.flatten(
                [datadict1['meta'][k]] + [datadict2['meta'][k]])

    if 'plot_varconverter' in datadict1:
        new_data['plot_varconverter'] = datadict1['plot_varconverter']
    assert datadict1['paraminfo'] == datadict2['paraminfo']
    new_data['paraminfo'] = datadict1['paraminfo']
    # print('datadict1 paraminfo ', datadict1['paraminfo'])
    # print('interp_rg_join' in datadict1['paraminfo'], 'rg' in datadict1, datadict1['paraminfo']['interp_rg_join'] == True, datadict1['paraminfo']['interp_rg_join'] in ["True", "true"])
    if 'interp_rg_join' in datadict1['paraminfo'] \
            and 'rg' in datadict1 \
            and (datadict1['paraminfo']['interp_rg_join'] == True \
                 or datadict1['paraminfo']['interp_rg_join'] in ["True", "true"]):
        # experimental feature to interpolate the rg variable of the second
        # data container
        # print('inside funct', datadict1['rg'].shape, datadict2['rg'].shape, np.allclose(datadict1['rg'], datadict2['rg']))
        if datadict1['rg'].shape != datadict2['rg'].shape \
                or not np.allclose(datadict1['rg'], datadict2['rg']):
            logger.info("interp_rg_join set for {} {}".format(datadict1["system"], datadict1['name']))
            datadict2 = interpolate2d(datadict2, new_range=datadict1['rg'])
            logger.info("Ranges of {} {} have been interpolated. (".format(datadict1["system"], datadict1['name']))

    if container_type == ['time', 'aux'] \
            and datadict1['var'].shape[-1] != datadict2['var'].shape[0] \
            and not (datadict1['var'].shape[-1] == 1 and datadict2['var'].shape[-1] == 1):
        # catch the case, when limrad loads differnet ranges
        size_left = datadict1['var'].shape
        size_right = datadict2['var'].shape
        if size_left[-1] > size_right[0]:
            # the left array is larger => expand the right one
            delta = size_left[-1] - size_right[0]
            datadict2['var'] = np.pad(datadict2['var'], (0, delta), 'constant', constant_values=0)
            datadict2['mask'] = np.pad(datadict2['mask'], (0, delta), 'constant', constant_values=True)
        elif size_left[-1] < size_right[0]:
            # the right array is larger => expand the left one
            delta = size_right[0] - size_left[-1]
            dim_to_pad = (0, delta) if len(size_left) == 1 else ((0, 0), (0, delta))
            datadict1['var'] = np.pad(datadict1['var'], dim_to_pad, 'constant', constant_values=0)
            datadict1['mask'] = np.pad(datadict1['mask'], dim_to_pad, 'constant', constant_values=True)
        logger.warning("needed to modify aux val {} {} {} {} {}".format(
            datadict2["system"], datadict2['name'],
            datadict1['dimlabel'], size_left, size_right))

    if container_type == ['time', 'range'] \
            or container_type == ['time', 'range', 'vel'] \
            or container_type == ['time', 'range', 'dict']:
        assert datadict1['rg_unit'] == datadict2['rg_unit']
        new_data['rg_unit'] = datadict1['rg_unit']
        assert np.allclose(datadict1['rg'], datadict2['rg']), (datadict1['rg'], datadict2['rg'])
    if 'colormap' in datadict1 or 'colormap' in datadict2:
        assert datadict1['colormap'] == datadict2['colormap'], \
                "colormaps not equal {} {}".format(datadict1['colormap'], datadict2['colormap'])
        new_data['colormap'] = datadict1['colormap']
    if 'vel' in container_type:
        assert np.all(datadict1['vel'] == datadict2['vel']), "vel coordinate arrays not equal"
        new_data['vel'] = datadict1['vel']

    if 'var_definition' in datadict1:
        if datadict1['var_definition'] != datadict2['var_definition']:
            logger.warning('var_definition {} {}'.format(
                str(datadict1['var_definition']), str(datadict2['var_definition'])))
        # assert np.all(datadict1['var_definition'] == datadict2['var_definition']), "var_definition arrays not equal"
        new_data['var_definition'] = datadict1['var_definition']
    assert datadict1['var_unit'] == datadict2['var_unit']
    new_data['var_unit'] = datadict1['var_unit']
    assert datadict1['var_lims'] == datadict2['var_lims']
    new_data['var_lims'] = datadict1['var_lims']
    assert datadict1['system'] == datadict2['system']
    new_data['system'] = datadict1['system']
    assert datadict1['name'] == datadict2['name']
    new_data['name'] = datadict1['name']
    # assert datadict1['plot_varconverter'] == datadict2['plot_varconverter']
    # new_data['plot_varconverter'] = datadict1['plot_varconverter']
    logger.debug(new_data['dimlabel'])
    logger.debug(new_data['paraminfo'])

    if container_type == ['time', 'range'] \
            or container_type == ['time', 'range', 'vel'] \
            or container_type == ['time', 'range', 'cat'] \
            or container_type == ['time', 'range', 'dict']:
        new_data['rg'] = datadict1['rg']
        new_data['ts'] = np.hstack((datadict1['ts'], datadict2['ts']))
        new_data['var'] = np.vstack((datadict1['var'], datadict2['var']))
        new_data['mask'] = np.vstack((datadict1['mask'], datadict2['mask']))
        # print(new_data['ts'].shape, new_data['rg'].shape, new_data['var'].shape)
    elif container_type == ['time', 'aux']:
        new_data['ts'] = np.hstack((datadict1['ts'], datadict2['ts']))
        new_data['var'] = np.vstack((datadict1['var'], datadict2['var']))
        new_data['mask'] = np.vstack((datadict1['mask'], datadict2['mask']))
    else:
        new_data['ts'] = np.hstack((datadict1['ts'], datadict2['ts']))
        new_data['var'] = np.hstack((datadict1['var'], datadict2['var']))
        new_data['mask'] = np.hstack((datadict1['mask'], datadict2['mask']))

    return new_data


def interpolate1d(data, mask_thres=0.0, **kwargs):
    """same as interpolate2d but for 1d containers (time or range dimension must be len 1)

    Args:
        data: larda data container to be interpolated in its 1d-dimension
        **kwargs:
        new_time (for interpolation in time dimension) : new time vector to which data should be interpolated
        new_range (for interpolation in range dimension): new range vector to which data should be interpolated

    Returns:
        data_container
    """
    var = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    if data['dimlabel'] == ['time', 'range']:
        assert len(data['rg']) == 1 or len(data['ts']) == 1, "wrong data dimension."
        if len(data['rg']) == 1:
            interp_dim = 'time'
        else:
            interp_dim = 'range'
    else:
        interp_dim = data['dimlabel'][0]

    if interp_dim == 'time':
        vector = data['ts']
        assert "new_time" in kwargs, "have to supply new_time kwarg for interpolation in time"
        xnew = kwargs['new_time']
    elif interp_dim == 'range':
        vector = data['rg']
        assert "new_range" in kwargs, "have to supply new_range kwarg for interpolation in rg dimension"
        xnew = kwargs['new_range']
    var = var.squeeze()
    interp_var = scipy.interpolate.interp1d(vector, var, fill_value="extrapolate")
    interp_mask = scipy.interpolate.interp1d(vector, data['mask'].squeeze(), fill_value="extrapolate")
    new_var = interp_var(xnew)
    # extrapolation is often erroneous
    new_mask = interp_mask(xnew) > mask_thres
    interp_data = {**data}
    new_mask = np.logical_or(np.logical_or(xnew < min(vector), xnew > max(vector)), new_mask)
    if 'ts' in data: interp_data['ts'] = data['ts'] if len(data['ts']) == 1 else xnew
    if 'rg' in data: interp_data['rg'] = data['rg'] if len(data['rg']) == 1 else xnew
    interp_data['var'] = new_var
    interp_data['mask'] = new_mask
    return interp_data


def interpolate2d(data, mask_thres=0.1, **kwargs):
    """interpolate timeheight data container

    Args:
        mask_thres (float, optional): threshold for the interpolated mask
        **new_time (np.array): new time axis
        **new_range (np.array): new range axis
        **method (str): if not given, use scipy.interpolate.RectBivariateSpline
        valid method arguments:
            'linear' - scipy.interpolate.interp2d
            'nearest' - scipy.interpolate.NearestNDInterpolator
            'rectbivar' (default) - scipy.interpolate.RectBivariateSpline

    Returns:
        data_container
    """
    not_asc = np.where(np.diff(data['ts']) < 0.01)[0]
    print('not ascending ', not_asc)
    if len(not_asc) > 0:
        print('ts ', [h.ts_to_dt(ts) for ts in data['ts'][not_asc[0]-1:not_asc[0]+3]])
        data = {**data}
        print(data['var'].shape)
        data['ts'] = np.delete(data['ts'], not_asc)
        data['var'] = np.delete(data['var'], not_asc, axis=0)
        data['mask'] = np.delete(data['mask'], not_asc, axis=0)
        print(data['var'].shape)
    

    var = data['var'].copy()
    
    # var = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    # logger.debug('var min {}'.format(data['var'][~data['mask']].min()))
    method = kwargs['method'] if 'method' in kwargs else 'rectbivar'
    args_to_pass = {}
    if method == 'rectbivar':
        kx, ky = 1, 1
        interp_var = scipy.interpolate.RectBivariateSpline(data['ts'], data['rg'], var, kx=kx, ky=ky)
        interp_mask = scipy.interpolate.RectBivariateSpline(data['ts'], data['rg'], data['mask'].astype(np.float), kx=kx, ky=ky)
        args_to_pass["grid"] = True
    elif method == 'linear1d':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = scipy.interpolate.LinearNDInterpolator(points, var.flatten(), fill_value=-999.0)
        interp_mask = scipy.interpolate.LinearNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    elif method == 'linear':
        ts = np.reshape(np.repeat(data['ts'], len(data['rg'])), var.shape)
        rg = np.reshape(np.tile(data['rg'], len(data['ts'])), var.shape)
        nanmask = np.isfinite(var)
        interp_var = scipy.interpolate.interp2d(ts[nanmask], rg[nanmask], var[nanmask])
        interp_mask = scipy.interpolate.interp2d(data['ts'], data['rg'], np.transpose(data['mask']).astype(np.float))
    elif method == 'nearest':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = scipy.interpolate.NearestNDInterpolator(points, var.flatten())
        interp_mask = scipy.interpolate.NearestNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    else:
        raise ValueError('Unknown Interpolation Method', method)

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']

    if method in ["nearest", "linear1d"]:
        new_points = np.array(list(zip(np.repeat(new_time, len(new_range)), np.tile(new_range, len(new_time)))))
        new_var = interp_var(new_points).reshape((len(new_time), len(new_range)))
        new_mask = interp_mask(new_points).reshape((len(new_time), len(new_range)))
    else:
        new_var = interp_var(new_time, new_range, **args_to_pass)
        new_mask = interp_mask(new_time, new_range, **args_to_pass)

    # print('new_mask', new_mask)
    new_mask[new_mask > mask_thres] = 1
    new_mask[new_mask < mask_thres] = 0
    # print('new_mask', new_mask)

    # print(new_var.shape, new_var)
    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = new_var if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_var)
    interp_data['mask'] = new_mask if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_mask)
    logger.info("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, new_var.shape, new_mask.shape))

    return interp_data


def combine(func, datalist, keys_to_update, **kwargs):
    """apply a func to the variable

    Args:
        func: a function that takes [datacontainer1, datacontainer2, ..]
            as given input (order as given in datalist) and returns
            var, mask
        datalist: list of data containers or single data container
        keys_to_update: dictionary of keys to update

    Returns:
        new data container
    """

    if type(datalist) == list and len(datalist) > 1:
        assert np.all(datalist[0]['rg'] == datalist[1]['rg'])
        assert np.all(datalist[0]['ts'] == datalist[1]['ts'])

    # use the first dict as the base
    new_data = {**datalist[0]} if type(datalist) == list else {**datalist}
    new_data.update(keys_to_update)
    if not 'meta' in new_data:
        new_data['meta'] = {}

    new_data['var'], new_data['mask'] = func(datalist)
    if type(datalist) == list:
        new_data['meta']['contianer_history'] = {
            'filename': [e['filename'] for e in datalist],
            'paraminfo': [e['paraminfo'] for e in datalist],
        }
    else:
        new_data['meta']['container_history'] = {'filename': datalist['filename'],
                               'paraminfo': datalist['paraminfo']}

    return new_data


def slice_container(data, value={}, index={}):
    """slice a data_container either by values or indices (or combination of both)

    using on :py:func:`pyLARDA.helpers.argnearest`

    .. code::

        slice_container(data, value={'time': [timestamp1], 'range': [4000, 5000]})
        # or
        slice_container(data, value={'time': [timestamp1, timestamp2], 'range': [4000, 5000]})
        # or
        slice_container(data, index={'time': [10, 20], 'range': [5, 25]})
        #or
        slice_container(data, value={'time': [timestamp1, timestamp2]},
                              index={'range': [5, 25]})

    Args:
        value (dict): slice by value of coordinate axis
        index (dict): slice by index of axis

    Returns:
        a sliced container
    """
    dim_to_coord_array = {'time': 'ts', 'range': 'rg', 'vel': 'vel'}

    if "dict" == data["dimlabel"][-1]:
        data["dimlabel"] = data['dimlabel'][:-1]
    # setup slicer
    sliced_data = {**data}
    slicer_dict = {}
    for dim in data['dimlabel']:
        if dim in value:
            bounds = [h.argnearest(data[dim_to_coord_array[dim]], v) for v in value[dim]]
            assert bounds[0] < data[dim_to_coord_array[dim]].shape[0], \
                "lower bound above data top"
            slicer_dict[dim] = slice(*bounds) if len(bounds) > 1 else bounds[0]
        elif dim in index:
            slicer_dict[dim] = slice(*index[dim]) if len(index[dim]) > 1 else index[dim][0]
            assert index[dim][0] < data[dim_to_coord_array[dim]].shape[0], \
                "lower bound above data top"
        else:
            slicer_dict[dim] = slice(None)

    logger.debug("slicer dict {}".format(slicer_dict))
    new_dimlabel = []
    # slice the coordinate arrays
    for dim in data['dimlabel']:
        coord_name = dim_to_coord_array[dim]
        sliced_data[coord_name] = data[coord_name][slicer_dict[dim]]
        # print(dim, sliced_data[coord_name].shape, sliced_data[coord_name])
        # print(type(sliced_data[coord_name]))
        if type(sliced_data[coord_name]) in [np.ndarray, np.ma.core.MaskedArray]:
            if sliced_data[coord_name].shape[0] > 1:
                new_dimlabel.append(dim)
            elif sliced_data[coord_name].shape[0] == 1:
                sliced_data[coord_name] = sliced_data[coord_name][0]
            else:
                raise IndexError('Error while slicing! Time or range frame to small for slicing!')
    logger.debug("new_dimlabel {}".format(new_dimlabel))
    sliced_data['dimlabel'] = new_dimlabel
    # actual slicing the variable
    slicer = tuple([slicer_dict[dim] for dim in data['dimlabel']])
    sliced_data['var'] = data['var'][slicer]
    sliced_data['mask'] = data['mask'][slicer]
    if isinstance(sliced_data['var'], np.ma.MaskedArray) or isinstance(sliced_data['var'], np.ndarray):
        logger.info('sliced {} to shape {}'.format(data['var'].shape, sliced_data['var'].shape))
    return sliced_data


def select_closest(data, ts):
    """Select closest time steps from larda container to given time steps and replaces original time steps
    with given ones
    Useful when plotting flag data, as interpolation does not work with categorical data

    Args:
        data: larda container
        ts (ndarray): array with unix time steps

    Returns: 
        larda container with time steps closest to the given ones
    """
    in_ts = data['ts']
    ts_list = list()
    for t in ts:
        id_diff_min = h.argnearest(in_ts, t)  # find index of nearest time step to input time step
        ts_list.append(id_diff_min)  # append index to list
    data_new = h.put_in_container(data['var'][ts_list], data, ts=ts, mask=data['mask'][ts_list])

    return data_new


def plot_timeseries(data, **kwargs):
    """plot a timeseries data container

   Args:
       data (dict): data container
       **time_interval (list dt): constrain plot to this dt
       **z_converter (string): convert var before plotting
               use eg 'lin2z' or 'log'
       **var_converter (string): alternate name for the z_converter
       **fig_size (list): size of figure, default is ``[10, 5.7]``
       **linewidth (float): controls the line width
       **alpha (float): controls transparency between [0, 1]
       **label (string, Bool): True, label the data automatically, otherwise use string
       **time_diff_jumps (length of time difference between time step required so that it is recognized as a 'jump')

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
   """
    assert data['dimlabel'] == ['time'], 'wrong plot function for {}'.format(data['dimlabel'])
    logger.warning("DeprecationWarning: use plot_timeseries2 instead")

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    if 'label' in kwargs and kwargs['label']:
        label_str = data['system'] + data['variable_name']
    elif 'label' in kwargs and kwargs['label']:
        label_str = kwargs['label']
    else:
        label_str = ''

    lw = kwargs['linewidth'] if 'linewidth' in kwargs else 1.5
    al = kwargs['alpha'] if 'alpha' in kwargs else 1.0

    time_list = data['ts']
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    # this is the last valid index
    var = var.filled(-999)
    if 'time_diff_jumps' in kwargs:
        td_jumps = kwargs['time_diff_jumps']
    else:
        td_jumps = 60
    jumps = np.where(np.diff(time_list) > td_jumps)[0]

    for ind in jumps[::-1].tolist():
        logger.debug("jump at {} {}".format(ind, dt_list[ind - 1:ind + 2]))
        # and modify the dt_list
        dt_list.insert(ind + 1, dt_list[ind] + datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind + 1, -999, axis=0)

    var = np.ma.masked_equal(var, -999)

    fig, ax = plt.subplots(1, figsize=fig_size)
    vmin, vmax = data['var_lims']
    logger.debug("varlims {} {}".format(vmin, vmax))
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            # plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            ax.set_yscale('log')
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)

    ax.plot(dt_list, var, linewidth=lw, alpha=al, label=label_str)

    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    else:
        ax.set_xlim([dt_list[0], dt_list[-1]])
    ax.set_ylim([vmin, vmax])

    # ax.set_ylim([height_list[0], height_list[-1]])
    # ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    # ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time [UTC]", fontweight=fontweight, fontsize=fontsize)

    ylabel = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    ax.set_ylabel(ylabel, fontweight=fontweight, fontsize=fontsize)

    time_extend = dt_list[-1] - dt_list[0]
    logger.debug("time extend {}".format(time_extend))
    ax = _set_xticks_and_xlabels(ax, time_extend)

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)

    return fig, ax


def plot_profile(data, **kwargs):
    """plot a profile data container

    Args:
        data (dict): data container
        **range_interval (list): constrain plot to this ranges
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
        **var_converter (string): alternate name for the z_converter
        **fig_size (list): size of figure, default is ``[4, 5.7]``

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    assert data['dimlabel'] == ['range'], 'wrong plot function for {}'.format(data['dimlabel'])

    var = np.ma.masked_where(data['mask'], data['var']).copy()
    # this is the last valid index
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [4, 5.7]
    fig, ax = plt.subplots(1, figsize=fig_size)
    vmin, vmax = data['var_lims']
    logger.debug("varlims {} {}".format(vmin, vmax))
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            # plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            ax.set_xscale('log')
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)

    ax.plot(var, data['rg'], color='darkred', label=data['paraminfo']['location'])

    if 'range_interval' in kwargs.keys():
        ax.set_ylim(kwargs['range_interval'])
    else:
        ax.set_ylim([data['rg'][0], data['rg'][-1]])
    ax.set_xlim([vmin, vmax])

    ylabel = 'Height [{}]'.format(data['rg_unit'])
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    xlabel = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    ax.set_xlabel(xlabel, fontweight='semibold', fontsize=15)

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14,
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    fig.tight_layout()

    return fig, ax


def plot_timeheight(data, fig=None, ax=None, **kwargs):
    """plot a timeheight data container

    Args:
        data (dict): data container
        **time_interval (list dt): constrain plot to this dt
        **range_interval (list float): constrain plot to this ranges
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
        **var_converter (string): alternate name for the z_converter
        **contour: add a countour
        **fig_size (list): size of figure, default is ``[10, 5.7]``
        **zlim (list): set vmin and vmax of color axis
        **title: True/False or string, True will auto-generate title
        **rg_converter: True/false, True will convert from "m" to "km"
        **time_diff_jumps: default is 60
        **fig: add to an existing figure
        **ax: add to an existing axis
        **cbar: removes cbar if False
        **mask

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    logger.warning("DeprecationWarning: use plot_timeheight2 instead")
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    # check for availability of dimensions in larda container or xarray object
    if hasattr(data, 'coords'):
        assert data.coords.dims == ('ts', 'rg'), f'attribute error, check coords ... wrong plot function for {data}'
        mask = kwargs['mask'] if 'mask' in kwargs else data.mask.values
        var = data.values
        name = data.name
        colormap_name = kwargs['colormap'] if 'colormap' in kwargs else data.colormap
        rg_unit = data.attrs['rg_unit']
        var_unit = data.attrs['var_unit']
        system = data.system
    else:
        assert data['dimlabel'] == ['time', 'range'], f'wrong plot function for {data["dimlabel"]}'
        mask = data['mask']
        var = data['var']
        name = data['name']
        colormap_name = data['colormap']
        rg_unit = data['rg_unit']
        var_unit = data['var_unit']
        system = data["system"]

    time_list = data['ts']
    range_list = data['rg'] / 1000.0 if 'rg_converter' in kwargs and kwargs['rg_converter'] else data['rg']

    var = np.ma.masked_where(mask, var).copy()

    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]

    # this is the last valid index
    var = var.astype(np.float32).filled(-999)
    if 'time_diff_jumps' in kwargs:
        td_jumps = kwargs['time_diff_jumps']
    else:
        td_jumps = 60
    jumps = np.where(np.diff(time_list) > td_jumps)[0]
    for ind in jumps[::-1].tolist():
        logger.debug("masked jump at {} {}".format(ind, dt_list[ind - 1:ind + 2]))
        # and modify the dt_list
        dt_list.insert(ind + 1, dt_list[ind] + datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind + 1, np.full(range_list.shape, -999), axis=0)

    var = np.ma.masked_equal(var, -999)
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]

    fraction_color_bar = 0.13

    # hack for categorial plots; currently only working for cloudnet classification
    is_classification = name in ['CLASS', 'CLASS_v2', 'detection_status', 'voodoo_classification', 'CLOUDNET_class', 'target_classification',
                                 'voodoo_classification_post', 'voodoo_class_raw_nosmoothing']
    if is_classification:
        vmin, vmax = [-0.5, len(VIS_Colormaps.categories[colormap_name]) - 0.5]
        # make the figure a littlebit wider and
        # use more space for the colorbar
        fig_size[0] = fig_size[0] + 1.25
        fraction_color_bar = 0.23
    elif 'zlim' in kwargs:
        vmin, vmax = kwargs['zlim']
    elif len(data['var_lims']) == 2:
        vmin, vmax = data['var_lims']
    else:
        vmin, vmax = np.min(var), np.max(var)

    logger.debug("varlims {} {}".format(vmin, vmax))
    plotkwargs = {}
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if colormap_name in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap_name]
    else:
        colormap = colormap_name

    if not (fig and ax):
        fig, ax = plt.subplots(1, figsize=fig_size)

    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]),
                           range_list[:],
                           np.transpose(var[:, :]),
                           cmap=colormap,
                           vmin=vmin, vmax=vmax,
                           **plotkwargs
                           )

    if 'contour' in kwargs and bool(kwargs['contour']):
        cdata = kwargs['contour']['data']

        if isinstance(cdata, dict):
            cdata_var = cdata['var']
        else:
            cdata_var = cdata.values

        assert len(cdata) > 1, 'Contour data empty!'
        if 'rg_converter' in kwargs and kwargs['rg_converter']:
            cdata_rg = np.divide(cdata['rg'], 1000.0)
        else:
            cdata_rg = cdata['rg']

        dt_c = [datetime.datetime.utcfromtimestamp(time) for time in cdata['ts']]
        if 'levels' in kwargs['contour']:
            cont = ax.contour(dt_c, cdata_rg,
                              np.transpose(cdata_var),
                              kwargs['contour']['levels'],
                              linestyles='dashed', colors='black', linewidths=0.75
                              )
        else:
            cont = ax.contour(dt_c, cdata_rg,
                              np.transpose(cdata_var),
                              linestyles='dashed', colors='black', linewidths=0.75)

        ax.clabel(cont, fontsize=fontsize, inline=1, fmt='%1.1f°C', )

    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    if 'range_interval' in kwargs.keys():
        if 'rg_converter' in kwargs and kwargs['rg_converter']:
            ax.set_ylim(np.divide(kwargs['range_interval'], 1000.0))
        else:
            ax.set_ylim(kwargs['range_interval'])

    ylabel = 'Height [{}]'.format(rg_unit)
    if 'rg_converter' in kwargs and kwargs['rg_converter']:
        ylabel = 'Height [km]'

    ax.set_xlabel("Time [UTC]", fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontweight=fontweight, fontsize=fontsize)

    if 'zlabel' in kwargs and kwargs['zlabel'] == "":
        z_string = ""
    else:
        z_string = "{} {} [{}]".format(system, name, var_unit)

    cbar_flag = False if 'cbar' in kwargs and kwargs['cbar'] == False else True
    if cbar_flag:
        cbar = fig.colorbar(pcmesh, ax=ax, fraction=fraction_color_bar, pad=0.025)
        cbar.ax.set_ylabel(z_string, fontweight=fontweight, fontsize=fontsize)
        if is_classification:
            categories = VIS_Colormaps.categories[colormap_name]
            cbar.set_ticks(list(range(len(categories))))
            cbar.ax.set_yticklabels(categories)

        cbar.ax.tick_params(axis='both', which='major', labelsize=labelsize, width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
        if is_classification:
            cbar.ax.tick_params(labelsize=11)
            fig_size[0] = fig_size[0] - 1.25  # change back to original

    time_extend = dt_list[-1] - dt_list[0]
    logger.debug("time extend {}".format(time_extend))
    ax = _set_xticks_and_xlabels(ax, time_extend)

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)

    if 'title' in kwargs and type(kwargs['title']) == str:
        ax.set_title(kwargs['title'], fontsize=fontsize)
    elif 'title' in kwargs and type(kwargs['title']) == bool:
        if kwargs['title'] == True:
            formatted_datetime = (h.ts_to_dt(data['ts'][0])).strftime("%Y-%m-%d")
            if not (h.ts_to_dt(data['ts'][0])).strftime("%d") == (h.ts_to_dt(data['ts'][-1])).strftime("%d"):
                formatted_datetime = formatted_datetime + ' to ' + (h.ts_to_dt(data['ts'][-1])).strftime("%d")
            ax.set_title(data['paraminfo']['location'] + ', ' +
                         formatted_datetime, fontsize=fontsize)

    plt.subplots_adjust(right=0.99)
    fig.tight_layout()
    return fig, ax


def plot_barbs_timeheight(u_wind, v_wind, *args, **kwargs):
    """barb plot for plotting of horizontal wind vector

    Args:
        u_wind (dict): u component of horizontal wind, m/s
        v_wind (dict): v component of horizontal wind, m/s
        args:
        *sounding_data: data container (dict) Wyoming radiosounding, m/s

        **range_interval: range interval to be plotted
        **fig_size: size of png (default is [10, 5.7])
        **all_data: True/False, default is False (plot only every third height bin)
        **z_lim: min/max velocity for plot (default is 0, 25 m/s)
        **labelsize: size of the axis labels (default 12)
        **barb_length: length of the barb (default 5)
        **flip_barb: bool to flip the barb for the SH  (default is false (=NH))
        **fig: existing matplotlib figure
        **ax: existing matplotlib axis

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    # Plotting arguments
    fig = kwargs['fig'] if 'fig' in kwargs else None
    ax = kwargs['ax'] if 'ax' in kwargs else None
    all_data = kwargs['all_data'] if 'all_data' in kwargs else False
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]
    labelsize = kwargs['labelsize'] if 'labelsize' in kwargs else 14
    flip_barb = kwargs['flip_barb'] if 'flip_barb' in kwargs else False
    barb_length = kwargs['barb_length'] if 'barb_length' in kwargs else 5
    fraction_color_bar = 0.13
    colormap = u_wind['colormap']
    zlim = kwargs['z_lim'] if 'z_lim' in kwargs else [0, 25]

    if not all_data:
        # mask 2 out of 3 height indices
        h_max = u_wind['rg'].size
        mask_index = np.sort(np.concatenate([np.arange(2, h_max, 3), np.arange(3, h_max, 3)]))
        u_wind['mask'][:, mask_index] = True
        v_wind['mask'][:, mask_index] = True

    # Arrange a grid for barb plot
    [base_height, top_height] = kwargs['range_interval'] if 'range_interval' in kwargs else [u_wind['rg'].min(),
                                                                                             u_wind['rg'].max()]
    time_list = u_wind['ts']
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    y, x = np.meshgrid(u_wind['rg'], matplotlib.dates.date2num(dt_list[:]))

    # Apply mask to variables
    u_var = np.ma.masked_where(u_wind['mask'], u_wind['var']).copy()
    v_var = np.ma.masked_where(v_wind['mask'], v_wind['var']).copy()
    u_var = np.ma.masked_where(u_var > 1000, u_var)
    v_var = np.ma.masked_where(v_var > 1000, v_var)

    # Derive wind speed in knots, 1m/s= 1.943844knots
    vel = np.sqrt(u_var ** 2 + v_var ** 2)
    u_knots = u_var * 1.943844
    v_knots = v_var * 1.943844

    # start plotting

    if not (fig and ax):
        fig, ax = plt.subplots(1, figsize=fig_size)

    if 'style' in kwargs and kwargs['style'] == 'LIMCUBE':
        steps = np.arange(0, 21, 1)
        cMap = plt.get_cmap('jet')
        cMap.set_bad(color='grey', alpha=1.)
        norm = matplotlib.colors.BoundaryNorm(steps, cMap.N)
        cp = ax.pcolormesh(x, y, vel, vmin=0, vmax=18, cmap=cMap, norm=norm)
        divider = make_axes_locatable(ax)
        cax0 = divider.append_axes("right", size="3%", pad=0.5)
        c_bar = fig.colorbar(cp, cax=cax0, ax=ax, ticks=steps[::2])
        c_bar.ax.tick_params(labelsize=12)
        ax.barbs(x, y, u_knots, v_knots, vel, length=4, pivot='middle')
        c_bar.set_label('m/s')
    else:
        barb_plot = ax.barbs(x, y, u_knots, v_knots, vel, rounding=False, cmap=colormap, clim=zlim,
                             sizes=dict(emptybarb=0), length=barb_length, flip_barb=flip_barb)

        c_bar = fig.colorbar(barb_plot, fraction=fraction_color_bar, pad=0.025)
        c_bar.set_label('Advection Speed [m/s]', fontsize=15)

    # Formatting axes and ticks
    ax.set_xlabel("Time [UTC]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    time_extend = dt_list[-1] - dt_list[0]
    logger.debug("time extent {}".format(time_extend))
    ax = _set_xticks_and_xlabels(ax, time_extend)

    assert u_wind['rg_unit'] == v_wind['rg_unit'], "u_wind and v_wind range units"
    ylabel = 'Height [{}]'.format(u_wind['rg_unit'])
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    c_bar.ax.tick_params(axis='both', which='major', labelsize=labelsize, width=2, length=4)
    c_bar.ax.tick_params(axis='both', which='minor', width=2, length=3)

    # add 10% to plot width to accommodate barbs
    x_lim = [matplotlib.dates.date2num(dt_list[0] - 0.1 * time_extend),
             matplotlib.dates.date2num(dt_list[-1] + 0.1 * time_extend)]
    y_lim = [base_height, top_height]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if 'text' in kwargs:
        ax.text(.015, .94, kwargs['text'],
                horizontalalignment='left', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.75)
                )

    # Check for sounding data
    if len(args) > 0:
        if type(args[0]) == dict:
            sounding_data = args[0]
            at_x, at_y = np.meshgrid(matplotlib.dates.date2num(h.ts_to_dt(sounding_data['time'])),
                                     sounding_data['range'])
            u_sounding = sounding_data['u_wind'] * 1.943844
            v_sounding = sounding_data['v_wind'] * 1.943844
            vel_sounding = sounding_data['speed']

            barb_plot.sounding = ax.barbs(at_x, at_y, u_sounding, v_sounding,
                                          vel_sounding, rounding=False, cmap=colormap, clim=zlim,
                                          sizes=dict(emptybarb=0), length=barb_length)

    plt.subplots_adjust(right=0.99)
    return fig, ax


def plot_scatter(data_container1, data_container2, identity_line=True, **kwargs):
    """scatter plot for variable comparison between two devices or variables

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        c_lim (list): limits of var used for color axis
        **identity_line (bool): plot 1:1 line if True
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **custom_offset_lines (float): plot 4 extra lines for given distance
        **info (bool): print slope, interception point and R^2 value
        **fig_size (list): size of the figure in inches
        **font_size (int): default: 15
        **font_weight (int): default: semibold
        **colorbar (bool): if True, add a colorbar to the scatterplot
        **color_by (dict): data container 3rd device
        **scale (string): 'lin' or 'log' --> if you get a ValueError from matplotlib.colors
                          try setting scale to lin, log does not work for negative values!
        **cmap (string) : colormap
        **formstring (string): format string to use as key word argument to fig.colorbar, overwrites the default values
                               for 'lin' or 'log' scale. E.g. use "%.2f" for displaying two digits
        **Nbins (int) : number of bins for histograms

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    logger.warning("DeprecationWarning: use plot_scatter2 instead")

    fig_size = np.repeat(min(kwargs['fig_size']), 2) if 'fig_size' in kwargs else [6, 6]
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    var1_tmp = data_container1
    var2_tmp = data_container2

    combined_mask = np.logical_or(var1_tmp['mask'], var2_tmp['mask'])
    colormap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    # convert var from linear unit with any converter given in helpers
    if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
        var1 = h.get_converter_array(kwargs['z_converter'])[0](var1_tmp['var'][~combined_mask].ravel())
        var2 = h.get_converter_array(kwargs['z_converter'])[0](var2_tmp['var'][~combined_mask].ravel())
    else:
        var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
        var2 = var2_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [np.nanmin(var1), np.nanmax(var1)]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [np.nanmin(var2), np.nanmax(var2)]
    fig_size[0] = fig_size[0] + 2 if 'colorbar' in kwargs and kwargs['colorbar'] else fig_size[0]
    try:
        Nbins = kwargs['Nbins'] if 'Nbins' in kwargs else int(round((np.nanmax(var1) - np.nanmin(var1)) /
                                                                    (2 * (np.nanquantile(var1, 0.75) -
                                                                          np.nanquantile(var1, 0.25)) * len(var1) ** (-1 / 3))))
    except OverflowError:
        print(f'var1 {var1_tmp["name"]}: len is {len(var1)}, '
              f'IQR is {np.nanquantile(var1, 0.75)} - {np.nanquantile(var1, 0.25)},'
              f'max is {np.nanmax(var1)}, min is {np.nanmin(var1)}')
        Nbins = 100
    # Freedman-Diaconis rule: h=2×IQR×n−1/3. number of bins is (max−min)/h, where n is the number of observations
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram

    # create histogram plot
    s, i, r, p, std_err = stats.linregress(var1, var2)
    H, xedges, yedges = np.histogram2d(var1, var2, bins=Nbins, range=[x_lim, y_lim])
    H = np.ma.masked_less_equal(H, 0)

    if 'color_by' in kwargs:
        print("Coloring scatter plot by {}...\n".format(kwargs['color_by']['name']))
        # overwrite H
        H = np.zeros(H.shape)
        var3 = kwargs['color_by']['var'][~combined_mask].ravel()
        # get the bins of the 2d histogram using digitize
        x_coords = np.digitize(var1, xedges)
        y_coords = np.digitize(var2, yedges)
        # find unique bin combinations = pixels in scatter plot

        # sort x and y coordinates using lexsort
        # lexsort sorts by multiple columns, first by y_coords then by x_coords

        newer_order = np.lexsort((x_coords, y_coords))
        x_coords = x_coords[newer_order]
        y_coords = y_coords[newer_order]
        var3 = var3[newer_order]
        first_hit_y = np.searchsorted(y_coords, np.arange(1, Nbins + 2))
        first_hit_y.sort()
        first_hit_x = [np.searchsorted(x_coords[first_hit_y[j]:first_hit_y[j + 1]], np.arange(1, Nbins + 2))
                       + first_hit_y[j] for j in np.arange(Nbins)]

        for x in range(Nbins):
            for y in range(Nbins):
                H[y, x] = np.nanmedian(var3[first_hit_x[x][y]: first_hit_x[x][y + 1]])

    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, figsize=fig_size)

    c_lim = kwargs['c_lim'] if 'c_lim' in kwargs else [1, round(np.nanmax(H), int(np.log10(max(np.nanmax(H), 10.))))]

    if 'scale' in kwargs and kwargs['scale'] == 'lin':
        formstring = "%.0f"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), vmin=c_lim[0], vmax=c_lim[1], cmap=colormap)
    else:
        formstring = "%.2E"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm(vmin=c_lim[0], vmax=c_lim[1]),
                             cmap=colormap)

    if 'info' in kwargs and kwargs['info']:
        ax.text(0.01, 0.93, 'slope = {:5.3f}\nintercept = {:5.3f}\nR^2 = {:5.3f}'.format(s, i, r ** 2),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontweight=fontweight, labelsize=fontsize)

    # helper lines (1:1), ...
    if identity_line: _add_identity(ax, color='salmon', ls='-')

    if 'custom_offset_lines' in kwargs:
        offset = np.array([kwargs['custom_offset_lines'], kwargs['custom_offset_lines']])
        for i in [-2, -1, 1, 2]: ax.plot(x_lim, x_lim + i * offset, color='salmon', linewidth=0.7, linestyle='--')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        # ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('{} {} [{}]'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel('{} {} [{}]'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    if 'colorbar' in kwargs and kwargs['colorbar']:
        cmap = copy(plt.get_cmap(colormap))
        cmap.set_under('white', 1.0)
        formstring = kwargs['formstring'] if 'formstring' in kwargs else formstring
        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8, format=formstring)
        if 'color_by' in kwargs:
            cbar.set_label(label="median {} [{}]".format(kwargs['color_by']['name'], kwargs['color_by']['var_unit']), fontweight=fontweight, fontsize=fontsize)
        else:
            cbar.set_label(label="frequency of occurrence", fontweight=fontweight, fontsize=fontsize)
        cbar.mappable.set_clim(c_lim)
        cbar.aspect = 50

    if 'title' in kwargs:
        if kwargs['title'] == True:
            ax.set_title(data_container1['paraminfo']['location'] +
                         h.ts_to_dt(data_container1['ts'][0]).strftime(" %Y-%m-%d %H:%M - ") +
                         h.ts_to_dt(data_container1['ts'][-1]).strftime("%Y-%m-%d %H:%M"), fontweight=fontweight, fontsize=fontsize)
        else:
            ax.set_title(kwargs['title'], fontweight=fontweight, fontsize=fontsize)

    plt.grid(b=True, which='both', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    # ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        cbar.ax.tick_params(axis='both', which='major', labelsize=labelsize, width=2, length=4)

    return fig, ax


def plot_frequency_of_occurrence(data, legend=True, **kwargs):
    """Frequency of occurrence diagram of a variable (number of occurrence for each range bin).
    x-axis is separated into n bins, default value for n = 100.

    Args:
        data (dict): container of Ze values
        **n_bins (integer): number of bins for reflectivity values (x-axis), default 100
        **x_lim (list): limits of x-axis, default: data['var_lims']
        **y_lim (list): limits of y-axis, default: minimum and maximum of data['rg']
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **range_offset (list): range values where chirp shift occurs
        **sensitivity_limit (np.array): 1-Dim array containing the minimum sensitivity values for each range
        **title (string): plot title string if given, otherwise not title
        **legend (bool): prints legend, default True

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    n_rg = data['rg'].size

    # create a mask for fill_value = -999. because numpy.histogram can't handle masked values properly
    var = copy(data['var'])
    var[data['mask']] = -999.0

    n_bins = kwargs['n_bins'] if 'n_bins' in kwargs else 100
    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else data['var_lims']
    v_lim = kwargs['v_lim'] if 'v_lim' in kwargs else [0.01, 20]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [data['rg'].min(), data['rg'].max()]

    # create bins of x and y axes
    x_bins = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins = data['rg']

    # initialize array
    H = np.zeros((n_bins - 1, n_rg))

    for irg in range(n_rg):
        # check for key word arguments
        nonzeros = copy(var[:, irg])[var[:, irg] != -999.0]
        if 'var_converter' in kwargs:
            kwargs['z_converter'] = kwargs['var_converter']
        if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
            nonzeros = h.get_converter_array(kwargs['z_converter'])[0](nonzeros)
            if kwargs['z_converter'] == 'lin2z': data['var_unit'] = 'dBZ'

        H[:, irg] = np.histogram(nonzeros, bins=x_bins, density=False)[0]

    H = np.ma.masked_equal(H, 0.0)

    # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
    cmap = copy(plt.get_cmap('viridis'))
    cmap.set_under('white', 1.0)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    pcol = ax.pcolormesh(x_bins, y_bins, H.T, vmin=v_lim[0], vmax=v_lim[1], cmap=cmap, label='histogram')

    cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8,
                        format='%2d')
    cbar.set_label(label="Frequencies of occurrence of {} values ".format(data['name']), fontweight='bold')
    cbar.aspect = 80

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('{} {} [{}]'.format(data['system'], data['name'], data['var_unit']), fontweight='bold')
    ax.set_ylabel('Height [{}]'.format(data['rg_unit']), fontweight='bold')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)

    if 'sensitivity_limit' in kwargs:
        sens_lim = kwargs['sensitivity_limit']
        if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
            sens_lim = h.get_converter_array(kwargs['z_converter'])[0](sens_lim)

        ax.plot(sens_lim, y_bins, linewidth=2.0, color='red', label='sensitivity limit')

    if 'range_offset' in kwargs and min(kwargs['range_offset']) <= max(y_lim):
        rg = kwargs['range_offset']
        ax.plot(x_lim, [rg[0]] * 2, linestyle='-.', linewidth=1, color='black', alpha=0.5, label='chirp shift')
        ax.plot(x_lim, [rg[1]] * 2, linestyle='-.', linewidth=1, color='black', alpha=0.5)

    if 'title' in kwargs: ax.set_title(kwargs['title'])

    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
    if legend: plt.legend(loc='upper left')
    fig.tight_layout()

    return fig, ax


def plot_foo_general(data_1, data_2, legend=True, **kwargs):
    """Frequency of occurrence diagram of a variable (number of occurrence binned by another variable).
    x-axis is separated into n bins, default value for n = 100.

    Args:
        data_1 (dict): container of e.g. Ze values
        data_2 (dict): container of e.g. velocity values
        **x_bins (integer): number of bins for dataset 2 values (x-axis), default 100
        **y_bins (integer): number of bins for dataset 1 values (y-axis), default 100
        **x_lim (list): limits of x-axis, default: data_2['var_lims']
        **y_lim (list): limits of y-axis, default: data_1['var_lims']
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **range_offset (list): range values where chirp shift occurs
        **sensitivity_limit (np.array): 1-Dim array containing the minimum sensitivity values for each range
        **title (string): plot title string if given, otherwise not title
        **legend (bool): prints legend, default True

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    #  Make sure the shapes of the two data sets are identical.
    assert data_1['var'].shape == data_2['var'].shape, "Data sets don't have the same shape."

    # create a mask for fill_value = -999. because numpy.histogram can't handle masked values properly
    var = copy(data_1['var'])
    var[data_1['mask']] = -999.0

    var_for_binning = copy(data_2['var'])
    var_for_binning[data_2['mask']] = -999.0

    xn_bins = kwargs['x_bins'] if 'x_bins' in kwargs else 100
    yn_bins = kwargs['y_bins'] if 'y_bins' in kwargs else 100

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else data_2['var_lims']
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else data_1['var_lims']

    # create bins of x and y axes
    x_bins = np.linspace(x_lim[0], x_lim[1], xn_bins)
    y_bins = np.linspace(y_lim[0], y_lim[1], yn_bins)

    # initialize array
    H = np.zeros((xn_bins - 1, yn_bins - 1))

    # loop over bins of var_to_bin
    for x in range(xn_bins - 1):
        it, ir = np.where(np.logical_and(var_for_binning > x_bins[x], var_for_binning < x_bins[x + 1]))
        # find index where var_to_bin is in the current bin
        # extract var for this index and convert it if needed
        nonzeros = copy(var[it, ir])[var[it, ir] != -999.0]
        if 'var_converter' in kwargs:
            kwargs['z_converter'] = kwargs['var_converter']
        if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
            nonzeros = h.get_converter_array(kwargs['z_converter'])[0](nonzeros)
            if kwargs['z_converter'] == 'lin2z': data_1['var_unit'] = 'dBZ'

        H[x, :] = np.histogram(nonzeros, bins=y_bins, density=True)[0]

    H = np.ma.masked_equal(H, 0.0)

    # create figure containing the frequency of occurrence of var over var_to_bin bins
    cmap = copy(plt.get_cmap('viridis'))
    cmap.set_under('white', 1.0)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    pcol = ax.pcolormesh(x_bins, y_bins, H.T, cmap=cmap, label='histogram')

    cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8)
    cbar.set_label(label="Normalized Frequency of occurrence of {} ".format(data_1['name']), fontweight='bold')
    cbar.aspect = 80

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('{} {} [{}]'.format(data_2['system'], data_2['name'], data_2['var_unit']), fontweight='bold')
    ax.set_ylabel('{} {} [{}]'.format(data_1['system'], data_1['name'], data_1['var_unit']), fontweight='bold')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)

    if 'title' in kwargs: ax.set_title(kwargs['title'])

    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
    if legend: plt.legend(loc='upper left')
    fig.tight_layout()

    return fig, ax


def _add_identity(axes, *line_args, **line_kwargs):
    """helper function for the scatter plot
    
    
    Returns:
        axes

    """
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def plot_spectra(data, *args, **kwargs):
    """Finds the closest match to a given point in time and height and plot Doppler spectra.

        Notes:
            The user is able to provide sliced containers, e.g.

            - one spectrum: ``data['dimlabel'] = ['vel']``
            - range spectrogram: ``data['dimlabel'] = ['range', 'vel']``
            - time spectrogram: ``data['dimlabel'] = ['time, 'vel']``
            - time-range spectrogram: ``data['dimlabel'] = ['time, 'range', 'vel']``

        Args:
            data (dict): data container
            *data2 (dict or numpy.ndarray): data container of a second device
            **z_converter (string): convert var before plotting use eg 'lin2z'
            **var_converter (string): alternate name for the z_converter
            **velmin (float): minimum x axis value
            **velmax (float): maximum x axis value
            **vmin (float): minimum y axis value
            **vmax (float): maximum y axis value
            **save (string): location where to save the pngs
            **fig_size (list): size of png, default is [10, 5.7]
            **mean (float): numpy array dimensions (time, height, 2) containing mean noise level for each spectra
                            in linear units [mm6/m3]
            **thresh (float): numpy array dimensions (time, height, 2) containing noise threshold for each spectra
                              in linear units [mm6/m3]
            **text (Bool): should time/height info be added as text into plot?
            **title (str or bool)
            **smooth (bool): if True, regular pyplot plot function is used (default is step)
            **alpha (float): triggers transparency of the line plot (not the bar plot), 0 <= alpha <= 1

        Returns:
            tuple with

            - fig (pyplot figure): contains the figure of the plot
              (for multiple spectra, the last fig is returned)
            - ax (pyplot axis): contains the axis of the plot
              (for multiple spectra, the last ax is returned)
        """

    fsz = 15
    velocity_min = -8.0
    velocity_max = 8.0
    annot = kwargs['text'] if 'text' in kwargs else True
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0

    n_time, n_height = data['ts'].size, data['rg'].size
    vel = data['vel'].copy()

    time, height, var, mask = h.reshape_spectra(data)

    velmin = kwargs['velmin'] if 'velmin' in kwargs else max(min(vel), velocity_min)
    velmax = kwargs['velmax'] if 'velmax' in kwargs else min(max(vel), velocity_max)

    smooth = kwargs['smooth'] if 'smooth' in kwargs else False

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]

    vmin = kwargs['vmin'] if 'vmin' in kwargs else data['var_lims'][0]
    vmax = kwargs['vmax'] if 'vmax' in kwargs else data['var_lims'][1]

    logger.debug("x-axis varlims {} {}".format(velmin, velmax))
    logger.debug("y-axis varlims {} {}".format(vmin, vmax))
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
        var = h.get_converter_array(kwargs['z_converter'])[0](var)

    name = kwargs['save'] if 'save' in kwargs else ''

    if len(args) > 0:
        if type(args[0]) == dict:
            data2 = args[0]
            vel2 = data2['vel'].copy()
            time2, height2, var2, mask2 = h.reshape_spectra(data2)
            if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
                var2 = h.get_converter_array(kwargs['z_converter'])[0](var2)
            second_data_set = True
    else:
        second_data_set = False

    # plot spectra
    ifig = 1
    n_figs = n_time * n_height

    for iTime in range(n_time):
        for iHeight in range(n_height):
            fig, ax = plt.subplots(1, figsize=fig_size)

            dTime = h.ts_to_dt(time[iTime])
            rg = height[iHeight]

            if annot:
                ax.text(0.01, 0.93,
                        '{} UTC  at {:.2f} m ({})'.format(dTime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], rg,
                                                          data['system']),
                        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            if not smooth:
                ax.step(vel, var[iTime, iHeight, :], color='royalblue', linestyle='-',
                        linewidth=2, label=data['system'] + ' ' + data['name'])
            else:
                ax.plot(vel, var[iTime, iHeight, :], color='royalblue', linestyle='-',
                        linewidth=2, alpha=alpha, label=data['system'] + ' ' + data['name'])

            # if a 2nd dict is given, assume another dataset and plot on top
            if second_data_set:
                # find the closest spectra to the first device
                iTime2 = h.argnearest(time2, time[iTime])
                iHeight2 = h.argnearest(height2, rg)

                dTime2 = h.ts_to_dt(time2[iTime2])
                rg2 = height2[iHeight2]

                if annot:
                    ax.text(0.01, 0.85,
                            '{} UTC  at {:.2f} m ({})'.format(dTime2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], rg2,
                                                              data2['system']),
                            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                if not smooth:
                    ax.step(vel2, var2[iTime2, iHeight2, :], color='darkred', linestyle='-',
                            linewidth=2, label=data2['system'] + ' ' + data2['name'])
                else:
                    ax.plot(vel2, var2[iTime2, iHeight2, :], color='darkred', linestyle='-',
                            linewidth=2, alpha=alpha, label=data2['system'] + ' ' + data2['name'])

            if 'mean' in kwargs or 'thresh' in kwargs:
                x1, x2 = vel[0], vel[-1]

                if 'mean' in kwargs and kwargs['mean'][iTime, iHeight] > 0.0:
                    mean = h.lin2z(kwargs['mean'][iTime, iHeight]) if kwargs['mean'].shape != () \
                        else h.lin2z(kwargs['mean'])
                    legendtxt_mean = 'mean noise floor =  {:.2f} '.format(mean)
                    ax.plot([x1, x2], [mean, mean], color='k', linestyle='--', linewidth=1, label=legendtxt_mean)

                if 'thresh' in kwargs and kwargs['thresh'][iTime, iHeight] > 0.0:
                    thresh = h.lin2z(kwargs['thresh'][iTime, iHeight]) if kwargs['thresh'].shape != () \
                        else h.lin2z(kwargs['thresh'])
                    legendtxt_thresh = 'noise floor threshold =  {:.2f} '.format(thresh)
                    ax.plot([x1, x2], [thresh, thresh], color='k', linestyle='-', linewidth=1, label=legendtxt_thresh)

            ax.set_xlim(left=velmin, right=velmax)
            ax.set_ylim(bottom=vmin, top=vmax)
            ax.set_xlabel('Doppler Velocity [m s$^{-1}$]', fontweight='semibold', fontsize=fsz)
            ax.set_ylabel('Reflectivity [dBZ]', fontweight='semibold', fontsize=fsz)
            ax.grid(linestyle=':')
            ax.tick_params(axis='both', which='major', labelsize=fsz)
            if 'title' in kwargs and type(kwargs['title']) == str:
                ax.set_title(kwargs['title'], fontsize=20)
            elif 'title' in kwargs and type(kwargs['title']) == bool:
                if kwargs['title'] == True:
                    formatted_datetime = dTime.strftime("%Y-%m-%d %H:%M")
                    ax.set_title("{}, {}, {} km".format(data['paraminfo']['location'], formatted_datetime,
                                                        str(round(rg) / 1000)),
                                 fontsize=20)
            # ax.tick_params(axis='both', which='minor', labelsize=8)

            ax.legend(fontsize=fsz)
            plt.tight_layout()

            if 'save' in kwargs:
                figure_name = name + '{}_{}_{:.0f}.png'.format(str(ifig).zfill(4),
                                                               dTime.strftime('%Y%m%d_%H%M%S_UTC'),
                                                               height[iHeight])
                fig.savefig(figure_name, dpi=100)
                print("   Saved {} of {} png to  {}".format(ifig, n_figs, figure_name))

            ifig += 1
            if ifig != n_figs + 1: plt.close(fig)

    return fig, ax


def plot_spectrogram(data, **kwargs):
    """Plot a time or height spectrogram

    The user is able to provide sliced containers, e.g.

    - range spectrogram: ``data['dimlabel'] = ['range', 'vel']``
    - time spectrogram: ``data['dimlabel'] = ['time, 'vel']``
    - time-range spectrogram: ``data['dimlabel'] = ['time, 'range', 'vel']``

    In the latter case, a height or time (value or index) must be provided
    at which the time / height spectrogram should be drawn

    Args:
        data: data container
        **index (dict): either {'time': time index} or {'range': range index}
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **fig_size (list): size of png, default is [10, 5.7]
        **v_lims (list): limits of Doppler velocity to be plotted

    Returns:
        tuple with

        - fig (pyplot figure): contains the figure of the plot
        - ax (pyplot axis): contains the axis of the plot
    """
    # Plotting parameters
    fsz = kwargs['font_size'] if 'font_size' in kwargs else 12
    fwgt = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]
    cbar_flag = kwargs['cbar'] if 'cbar' in kwargs else True
    colormap = data['colormap']
    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if colormap in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap]

    if 'rg_converter' in kwargs:
        data['rg'] = data['rg']/1000 if kwargs['rg_converter'] else data['rg']
        data['rg_unit'] = 'km' if kwargs['rg_converter'] else 'm'

    fraction_color_bar = 0.13

    n_time, n_height = data['ts'].size, data['rg'].size
    vel = data['vel'].copy()
    time, height, var, mask = h.reshape_spectra(data)
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        var = h.get_converter_array(kwargs['z_converter'])[0](var)
    var = np.ma.masked_where(mask, var)
    var = var.astype(np.float64).filled(-999)
    index = kwargs['index'] if 'index' in kwargs else ''

    # depending on dimensions of given data, decide if height or time spectrogram should be plotted
    # (1) dimensions ar time and height, then a h or t must be given
    if (n_height > 1) & (n_time > 1):
        assert 'index' in kwargs, "For time-height data container, you need to pass a time or height index to plot " \
                                  "the spectrogram, e.g. index={'time':5}"
        method = 'range_spec' if 'time' in index.keys() else 'time_spec' if 'height' in index.keys() else ''
        idx = index['time'] if method == 'range_spec' else index['height'] if method == 'time_spec' else ''
        time = time[idx] if method == 'range_spec' else time
        height = height[idx] if method == 'time_spec' else height
        var = var[:, idx, :] if method == 'time_spec' else var[idx, :, :] if method == 'range_spec' else var
    # (2) only time dimension
    elif (n_height > 1) & (n_time == 1):
        method = 'range_spec'
        var = np.squeeze(var)
    # (3) only height dimension
    elif (n_height == 1) & (n_time > 1):
        method = 'time_spec'
        var = np.squeeze(var)
        height = height[0]
    # (4) only one spectrum, Error
    assert not (n_height == 1) & (n_time == 1), 'Only one spectrum given.'
    assert method != '', 'Method not found. Check your index definition.'

    if method == 'range_spec':
        x_var = vel
        y_var = height
    elif method == 'time_spec':
        dt_list = [datetime.datetime.utcfromtimestamp(t) for t in list(time)]
        y_var = vel
        # identify time jumps > 60 seconds (e.g. MIRA scans)
        jumps = np.where(np.diff(list(time)) > 60)[0]
        for ind in jumps[::-1].tolist():
            # start from the end or stuff will be inserted in the beginning and thus shift the index
            logger.debug("masked jump at {} {}".format(ind, dt_list[ind - 1: ind + 2]))
            dt_list.insert(ind + 1, dt_list[ind] + datetime.timedelta(seconds=5))
            var = np.insert(var, ind + 1, np.full(vel.shape, -999), axis=0)
        var = np.transpose(var[:, :])
        x_var = matplotlib.dates.date2num(dt_list)

    var = np.ma.masked_equal(var, -999)
    # start plotting

    fig, ax = plt.subplots(1, figsize=fig_size)
    pcmesh = ax.pcolormesh(x_var, y_var, var[:, :], cmap=colormap, vmin=data['var_lims'][0], vmax=data['var_lims'][1])
    cbar = None
    if cbar_flag:
        pad = kwargs['bar_pad'] if 'bar_pad' in kwargs else 0.025
        cbar = fig.colorbar(pcmesh, fraction=fraction_color_bar, pad=pad)

    if 'v_lims' in kwargs.keys():
        if method == 'range_spec':
            ax.set_xlim(kwargs['v_lims'])
        elif method == 'time_spec':
            ax.set_ylim(kwargs['v_lims'])
    if method == 'range_spec':

        ax.set_xlabel('Velocity [m s$\\mathregular{^{-1}}$]', fontweight=fwgt, fontsize=fsz)
        ylabel = 'Range [{}]'.format(data['rg_unit'], fontsize=fsz)
        ax.set_ylabel(ylabel, fontweight='semibold', fontsize=fsz)
    elif method == 'time_spec':
        ax.set_ylabel('Velocity [m s$\\mathregular{^{-1}}$]', fontweight=fwgt, fontsize=fsz)
        ax.set_xlabel('Time [UTC]', fontweight=fwgt, fontsize=fsz)
        time_extend = dt_list[-1] - dt_list[0]
        ax = _set_xticks_and_xlabels(ax, time_extend)

    if 'title' in kwargs and kwargs['title']:
        ax.set_title("{} spectrogram at {} ".format(method.split('_')[0],
                                                    h.ts_to_dt(time).strftime('%d.%m.%Y %H:%M:%S') if method == 'range_spec'
                                                    else str(round(height)) + ' ' + data['rg_unit']),
                     fontsize=15, fontweight='semibold')
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=fsz, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    if cbar:
        cbar.ax.tick_params(axis='both', which='major', labelsize=fsz, width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
        if not ('bar' in kwargs and kwargs['bar'] == 'horizontal'):
            if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
                z_string = "{} {} [{}{}]".format(data["system"], data["name"], "dB",
                                                 data['var_unit'])
            else:
                z_string = ''
            cbar.ax.set_ylabel(z_string, fontweight=fwgt, fontsize=fsz)

        cbar.ax.minorticks_on()

    if 'grid' in kwargs and kwargs['grid'] == 'major':
        ax.grid(linestyle=':')

    if method == 'time_spec':
        time_extend = dt_list[-1] - dt_list[0]
        logger.debug("time extent {}".format(time_extend))
        ax = _set_xticks_and_xlabels(ax, time_extend)

    return fig, [ax, pcmesh]


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image path.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:, :, :3]
        output = img if i == 0 else concat_images(output, img)
    return output


def plot_ppi(data, azimuth, **kwargs):
    """plot a mira plan-position-indicator scan

    Args:
        data (dict): data_container holding the variable of the scan (Z, v, ..)
        azimuth (dict): data_container with the azimuth data
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **elv (float): elevation other than 75 deg
        **fig_size (list): size of png, default is [10, 5.7]

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    labelsize = 14
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 8]
    # if no elevation angle is supplied, set it to 75 degrees
    elv = kwargs['elv'] if 'elv' in kwargs else 75
    plotkwargs = {}
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    vmin, vmax = data['var_lims']

    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    colormap = kwargs['cmap'] if 'cmap' in kwargs else data['colormap']
    # spherical coordinates to kartesian
    ranges = data['rg']
    elv = elv * np.pi / 180.0
    # elevations = np.repeat(elv, len(ranges))
    azimuths = azimuth['var'] * np.pi / 180.0
    # elevations = np.transpose(np.repeat(elevations[:,np.newaxis], len(azimuths), axis = 1))
    azimuths = np.repeat(azimuths[:, np.newaxis], len(ranges), axis=1)
    ranges = np.tile(ranges, (len(data['var']), 1))
    x = ranges * np.sin(elv) * np.sin(azimuths)
    y = ranges * np.sin(elv) * np.cos(azimuths)
    fig, ax = plt.subplots(1, figsize=fig_size)
    mesh = ax.pcolormesh(x, y, var, cmap=colormap, vmin=vmin, vmax=vmax, **plotkwargs)
    ax.grid(linestyle=':')
    ax.set_xlabel('Horizontal distance [km]', fontsize=13)
    ax.set_ylabel('Horizontal distance [km]', fontsize=13)
    cbar = fig.colorbar(mesh, fraction=0.13, pad=0.05)

    if data['var_unit'] == "":
        z_string = "{} {}".format(data["system"], data["name"])
    else:
        z_string = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    cbar.ax.set_ylabel(z_string, fontweight='semibold', fontsize=15)

    return fig, ax


def plot_rhi(data, elv, **kwargs):
    """plot a mira range-height-indicator scan

    Args:
        data (dict): data_container holding the variable of the scan (Z, LDR, ..)
        elv (dict): data_container with the elevation data
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **fig_size (list): size of png, default is [10, 5.7]
        **title (str or bool)

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    labelsize = 14
    fig_size = kwargs['figsize'] if 'figsize' in kwargs else [10, 5.7]
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    vmin, vmax = data['var_lims']
    plotkwargs = {}
    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    colormap = data['colormap']

    ranges = np.tile(data['rg'], (len(data['var']), 1))
    elevations = np.repeat(elv['var'][:, np.newaxis], len(data['rg']), axis=1)
    h_distance = ranges * np.sin(elevations * np.pi / 180.0)
    v_distance = ranges * np.cos(elevations * np.pi / 180.0)
    fig, ax = plt.subplots(1, figsize=fig_size)
    mesh = ax.pcolormesh(v_distance / 1000.0, h_distance / 1000.0, var, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_xlim([np.min(v_distance) / 1000, np.max(v_distance) / 1000])
    ax.set_ylim([0, 8])
    ax.set_xlabel('Horizontal range [km]', fontsize=13)
    ax.set_ylabel('Height [km]', fontsize=13)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    cbar = fig.colorbar(mesh, fraction=0.13, pad=0.05)
    if data['var_unit'] == "" or data['var_unit'] == " ":
        z_string = "{} {}".format(data["system"], data["name"])
    else:
        z_string = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    cbar.ax.set_ylabel(z_string, fontweight='semibold', fontsize=15)

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize,
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.tick_params(axis='both', which='major', labelsize=labelsize,
                        width=2, length=4)
    cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

    if 'title' in kwargs and type(kwargs['title']) == str:
        ax.set_title(kwargs['title'], fontsize=20)
    elif 'title' in kwargs and type(kwargs['title']) == bool:
        if kwargs['title'] == True:
            formatted_datetime = (h.ts_to_dt(data['ts'][0])).strftime("%Y-%m-%d %H:%M")
            ax.set_title(data['paraminfo']['location'] + ', ' +
                         formatted_datetime, fontsize=20)
    return fig, ax


def remsens_limrad_quicklooks(container_dict, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import LogFormatter
    import matplotlib.colors as mcolors
    import time

    tstart = time.time()
    print('Plotting data...')

    site_name = container_dict['Ze']['paraminfo']['location']

    time_list = container_dict['Ze']['ts']
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    dt_list_2 = [datetime.datetime.utcfromtimestamp(time) for time in container_dict['LWP']['ts']]

    if 'timespan' in kwargs and kwargs['timespan'] == '24h':
        dt_lim_left = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0)
        dt_lim_right = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0) + datetime.timedelta(days=1)
    else:
        dt_lim_left = dt_list[0]
        dt_lim_right = dt_list[-1]

    range_list = container_dict['Ze']['rg'] * 1.e-3  # convert to km
    ze = h.lin2z(container_dict['Ze']['var']).copy().T
    mdv = container_dict['VEL']['var'].copy().T
    sw = container_dict['sw']['var'].copy().T
    ldr = container_dict['ldr']['var'].copy().T
    lwp = container_dict['LWP']['var'].copy()
    rr = container_dict['rr']['var'].copy()

    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [0, 12.0]

    # create figure

    fig, ax = plt.subplots(6, figsize=(13, 16))

    # reflectivity plot
    ax[0].text(.015, .87, 'Radar reflectivity factor', horizontalalignment='left',
               transform=ax[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[0].pcolormesh(dt_list, range_list, ze,
                          vmin=container_dict['Ze']['var_lims'][0],
                          vmax=container_dict['Ze']['var_lims'][1],
                          cmap=container_dict['Ze']['colormap'])
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax0, ax=ax[0])
    cbar.set_label('dBZ')
    ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... Ze')

    # mean doppler velocity plot
    ax[1].text(.015, .87, 'Mean Doppler velocity', horizontalalignment='left', transform=ax[1].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[1].pcolormesh(dt_list, range_list, mdv,
                          vmin=container_dict['VEL']['var_lims'][0],
                          vmax=container_dict['VEL']['var_lims'][1],
                          cmap=container_dict['VEL']['colormap'])
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax2, ax=ax[1])
    cbar.set_label('m/s')
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... mdv')

    # spectral width plot
    ax[2].text(.015, .87, 'Spectral width', horizontalalignment='left', transform=ax[2].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[2].pcolormesh(dt_list, range_list, sw,
                          norm=mcolors.LogNorm(vmin=0.1,
                                               vmax=container_dict['sw']['var_lims'][1]),
                          cmap=container_dict['sw']['colormap'])
    divider3 = make_axes_locatable(ax[2])
    cax3 = divider3.append_axes("right", size="3%", pad=0.3)
    formatter = LogFormatter(10, labelOnlyBase=False)
    cbar = fig.colorbar(cp, cax=cax3, ax=ax[2], format=formatter, ticks=[0.1, 0.2, 0.5, 1, 2])
    cbar.set_ticklabels([0.1, 0.2, 0.5, 1, 2])
    cbar.set_label('m/s')
    ax[2].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... sw')

    # linear depolarisation ratio plot
    colors1 = plt.cm.binary(np.linspace(0.5, 0.5, 1))
    colors2 = plt.cm.jet(np.linspace(0, 0, 178))
    colors3 = plt.cm.jet(np.linspace(0, 1, 77))
    colors = np.vstack((colors1, colors2, colors3))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    ax[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax[3].text(.015, .87, 'Linear depolarisation ratio', horizontalalignment='left',
               transform=ax[3].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[3].pcolormesh(dt_list, range_list, ldr, vmin=-100, vmax=0, cmap=mymap)
    divider4 = make_axes_locatable(ax[3])
    cax4 = divider4.append_axes("right", size="3%", pad=0.3)
    bounds = np.linspace(-30, 0, 500)
    cbar = fig.colorbar(cp, cax=cax4, ax=ax[3], boundaries=bounds, ticks=[-30, -25, -20, -15, -10, -5, 0])
    cbar.set_ticklabels([-30, -25, -20, -15, -10, -5, 0])
    cbar.set_label('dB')
    ax[3].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... ldr')

    # liquid water path plot
    ax[4].text(.015, .87, 'Liquid Water Path', horizontalalignment='left', transform=ax[4].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[4].bar(dt_list_2, lwp, width=0.001, color="blue", edgecolor="blue")
    ax[4].grid(linestyle=':')
    divider5 = make_axes_locatable(ax[4])
    cax5 = divider5.append_axes("right", size="3%", pad=0.3)
    cax5.axis('off')
    ax[4].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
    ax[4].set_ylabel('Liquid Water Path (g/$\mathregular{m^2}$)', fontsize=14)
    ax[4].set_xlim(left=dt_lim_left, right=dt_lim_right)
    ax[4].set_ylim(top=500, bottom=0)
    ax[4].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... lwp')

    # rain rate plot
    ax[5].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax[5].text(.015, .87, 'Rain rate', horizontalalignment='left', transform=ax[5].transAxes, fontsize=14,
               bbox=dict(facecolor='white', alpha=0.75))
    divider6 = make_axes_locatable(ax[5])
    cax6 = divider6.append_axes("right", size="3%", pad=0.3)
    cax6.axis('off')
    ax[5].grid(linestyle=':')
    cp = ax[5].bar(dt_list_2, rr, width=0.001, color="blue", edgecolor="blue")

    ax[5].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
    ax[5].axis([dt_list[0], dt_list[-1], 0, 10])
    ax[5].set_ylabel('Rain rate (mm/h)', fontsize=14)
    ax[5].set_xlim(left=dt_lim_left, right=dt_lim_right)
    ax[5].set_ylim(top=10, bottom=0)
    ax[5].set_xlabel('Time (UTC)')
    ax[5].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... rr')

    # duration of nc file for meteorological data calculation
    temp = container_dict['SurfTemp']['var'].copy()
    wind = container_dict['SurfWS']['var'].copy()
    tmin, tmax = min(temp) - 275.13, max(temp) - 275.13
    t_avg = np.mean(temp) - 275.13
    wind_avg = np.mean(wind)
    precip = np.mean(rr) * ((time_list[-1] - time_list[0]) / 3600.)

    txt = 'Meteor. Data: Avg. T.: {:.2f} °C;  Max. T.: {:.2f} °C;  Min. T.: {:.2f} °C;  ' \
          'Mean wind: {:.2f} m/s;  Total precip.: {:.2f} mm'.format(t_avg, tmax, tmin, wind_avg, precip)

    yticks = np.arange(plot_range[0] / 1000., plot_range[1] / 1000. + 1, 2)  # y-axis ticks

    for iax in range(4):
        ax[iax].grid(linestyle=':')
        ax[iax].set_yticks(yticks)
        ax[iax].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
        ax[iax].set_ylabel('Height (km)', fontsize=14)
        ax[iax].set_xlim(left=dt_lim_left, right=dt_lim_right)
        ax[iax].set_ylim(top=plot_range[1] / 1000., bottom=plot_range[0] / 1000.)
        ax[iax].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    fig.text(.5, .01, txt, ha="center", bbox=dict(facecolor='none', edgecolor='black'))
    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0.20)
    date_string = dt_lim_left.strftime("%Y%m%d")
    fig.suptitle("{}, {} (UTC), {}".format(container_dict['Ze']['system'], date_string, site_name), fontsize=20)
    # place in title needs to be adjusted

    print('plotting done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    return fig, ax


def remsens_limrad_polarimetry_quicklooks(container_dict, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import LogFormatter
    import matplotlib.colors as mcolors
    import time

    tstart = time.time()
    print('Plotting data...')

    site_name = container_dict['Ze']['paraminfo']['location']

    time_list = container_dict['Ze']['ts']
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    dt_list_2 = [datetime.datetime.utcfromtimestamp(time) for time in container_dict['LWP']['ts']]

    if 'timespan' in kwargs and kwargs['timespan'] == '24h':
        dt_lim_left = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0)
        dt_lim_right = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0) + datetime.timedelta(days=1)
    else:
        dt_lim_left = dt_list[0]
        dt_lim_right = dt_list[-1]

    range_list = container_dict['Ze']['rg'] * 1.e-3  # convert to km
    ze = h.lin2z(container_dict['Ze']['var']).T.copy()
    ldr = np.ma.masked_less_equal(container_dict['ldr']['var'].T, -999.0)
    zdr = np.ma.masked_less_equal(container_dict['ZDR']['var'].T, -999.0)
    rhv = np.ma.masked_less_equal(container_dict['RHV']['var'].T, -999.0)
    lwp = container_dict['LWP']['var'].copy()
    rr = container_dict['rr']['var'].copy()

    hmax = 12.0
    ticklen = 6.
    linewidth = 0.5

    cbar_ticklen = ticklen / 2.
    cbar_pad = 1.

    # create figure
    fig, ax = plt.subplots(6, figsize=(13, 16))

    # reflectivity plot
    ax[0].text(.015, .87, 'Radar reflectivity factor', horizontalalignment='left',
               transform=ax[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[0].pcolormesh(dt_list, range_list, ze,
                          vmin=container_dict['Ze']['var_lims'][0],
                          vmax=container_dict['Ze']['var_lims'][1],
                          cmap=container_dict['Ze']['colormap'])
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax0, ax=ax[0])
    cbar.set_label('dBZ')
    ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... Ze')

    # linear depolarisation ratio plot
    colors1 = plt.cm.binary(np.linspace(0.5, 0.5, 1))
    colors2 = plt.cm.jet(np.linspace(0, 0, 178))
    colors3 = plt.cm.jet(np.linspace(0, 1, 77))
    colors = np.vstack((colors1, colors2, colors3))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax[1].text(.015, .87, 'Linear depolarisation ratio', horizontalalignment='left',
               transform=ax[1].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[1].pcolormesh(dt_list, range_list, ldr, vmin=-100, vmax=0, cmap=mymap)
    divider4 = make_axes_locatable(ax[1])
    cax4 = divider4.append_axes("right", size="3%", pad=0.3)
    bounds = np.linspace(-30, 0, 500)
    cbar = fig.colorbar(cp, cax=cax4, ax=ax[1], boundaries=bounds, ticks=[-30, -25, -20, -15, -10, -5, 0])
    cbar.set_ticklabels([-30, -25, -20, -15, -10, -5, 0])
    cbar.set_label('dB')
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... ldr')

    # differential reflectivity plot
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, 1.6, 0.25), plt.get_cmap('jet').N)
    ax[2].text(.015, .87, 'Differential reflectivity', horizontalalignment='left', transform=ax[2].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[2].pcolormesh(dt_list, range_list, zdr,
                          vmin=container_dict['ZDR']['var_lims'][0],
                          vmax=container_dict['ZDR']['var_lims'][1],
                          cmap=container_dict['ZDR']['colormap'],
                          norm=norm)
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax2, ax=ax[2], ticks=[-0.5, 0, 0.5, 1.0, 1.5])
    cbar.set_label('dB')
    ax[2].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... ZDR')

    # correlation coefficient plot
    norm = matplotlib.colors.BoundaryNorm(np.arange(0.8, 1.01, 0.02), plt.get_cmap('jet').N)
    ax[3].text(.015, .87, 'Correlation coefficient $\\rho_{HV}$', horizontalalignment='left', transform=ax[3].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[3].pcolormesh(dt_list, range_list, rhv,
                          vmin=container_dict['RHV']['var_lims'][0],
                          vmax=container_dict['RHV']['var_lims'][1],
                          cmap=container_dict['RHV']['colormap'],
                          norm=norm)
    divider3 = make_axes_locatable(ax[3])
    cax3 = divider3.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax3, ax=ax[3])
    cbar.set_label('1')
    cax3.axes.tick_params(width=linewidth, length=cbar_ticklen, pad=cbar_pad)
    ax[3].axes.tick_params(axis='both', direction='inout', length=ticklen, width=linewidth)
    ax[3].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... RHV')

    # liquid water path plot
    ax[4].text(.015, .87, 'Liquid Water Path', horizontalalignment='left', transform=ax[4].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[4].bar(dt_list_2, lwp, width=0.001, color="blue", edgecolor="blue")
    ax[4].grid(linestyle=':')
    divider5 = make_axes_locatable(ax[4])
    cax5 = divider5.append_axes("right", size="3%", pad=0.3)
    cax5.axis('off')
    ax[4].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
    ax[4].set_ylabel('Liquid Water Path (g/$\mathregular{m^2}$)', fontsize=14)
    ax[4].set_xlim(left=dt_lim_left, right=dt_lim_right)
    ax[4].set_ylim(top=500, bottom=0)
    ax[4].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... lwp')

    # rain rate plot
    ax[5].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax[5].text(.015, .87, 'Rain rate', horizontalalignment='left', transform=ax[5].transAxes, fontsize=14,
               bbox=dict(facecolor='white', alpha=0.75))
    divider6 = make_axes_locatable(ax[5])
    cax6 = divider6.append_axes("right", size="3%", pad=0.3)
    cax6.axis('off')
    ax[5].grid(linestyle=':')
    cp = ax[5].bar(dt_list_2, rr, width=0.001, color="blue", edgecolor="blue")

    ax[5].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
    ax[5].axis([dt_list[0], dt_list[-1], 0, 10])
    ax[5].set_ylabel('Rain rate (mm/h)', fontsize=14)
    ax[5].set_xlim(left=dt_lim_left, right=dt_lim_right)
    ax[5].set_ylim(top=10, bottom=0)
    ax[5].set_xlabel('Time (UTC)')
    ax[5].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... rr')

    # duration of nc file for meteorological data calculation
    temp = container_dict['SurfTemp']['var'].copy()
    wind = container_dict['SurfWS']['var'].copy()
    tmin, tmax = min(temp) - 275.13, max(temp) - 275.13
    t_avg = np.mean(temp) - 275.13
    wind_avg = np.mean(wind)
    precip = np.mean(rr) * ((time_list[-1] - time_list[0]) / 3600.)

    txt = 'Meteor. Data: Avg. T.: {:.2f} °C;  Max. T.: {:.2f} °C;  Min. T.: {:.2f} °C;  ' \
          'Mean wind: {:.2f} m/s;  Total precip.: {:.2f} mm'.format(t_avg, tmax, tmin, wind_avg, precip)

    yticks = np.arange(0, hmax + 1, 2)  # y-axis ticks

    for iax in range(4):
        ax[iax].grid(linestyle=':')
        ax[iax].set_yticks(yticks)
        ax[iax].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
        ax[iax].set_ylabel('Height (km)', fontsize=14)
        ax[iax].set_xlim(left=dt_lim_left, right=dt_lim_right)
        ax[iax].set_ylim(top=hmax, bottom=0)
        ax[iax].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    fig.text(.5, .01, txt, ha="center", bbox=dict(facecolor='none', edgecolor='black'))
    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0.20)
    date_string = dt_lim_left.strftime("%Y%m%d")
    # place in title needs to be adjusted
    fig.suptitle("{}, {} (UTC), {}".format(container_dict['Ze']['system'], date_string, site_name), fontsize=20)

    print('plotting done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    return fig, ax


def plot_spectra_cwt(data, scalesmatr, iT=0, iR=0, legend=True, **kwargs):
    widths = kwargs['scales'] if 'scales' in kwargs else [0.0, 7.00]
    z_lim = kwargs['z_lim'] if 'z_lim' in kwargs else [scalesmatr.min(), scalesmatr.max()]
    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [data['vel'][0], data['vel'][-1]]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [-60, 20]

    colormap = kwargs['colormap'] if 'colormap' in kwargs else 'cloudnet_jet'
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.625]
    features = kwargs['features'] if 'features' in kwargs else np.nan

    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if colormap in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap]

    fontsize = 12

    time, height, var, mask = h.reshape_spectra(data)
    # convert from linear units to logarithic units
    vhspec = var.copy()

    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            # plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            pass
        else:
            signal_limrad = h.get_converter_array(kwargs['z_converter'])[0](vhspec)
    else:
        signal_limrad = vhspec

    if 'mira_spec' in kwargs:
        data2 = kwargs['mira_spec']
        vel2 = data2['vel'].copy()
        time2, height2, var2, mask2 = h.reshape_spectra(data2)
        if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
            signal_mira = h.get_converter_array(kwargs['z_converter'])[0](var2)
        second_data_set = True
    else:
        second_data_set = False

    cwtmatr_spec = scalesmatr
    rg = height[iR]

    # plot spectra
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size)

    ax[0].set_title('Doppler spectra, normalized and wavlet transformation\nheight: '
                    + str(round(height[iR], 2)) +
                    ' (m);  time: {} (UTC)'.format(h.ts_to_dt(time[iT]).strftime("%Y-%m-%d %H:%M:%S")),
                    fontweight='bold', fontsize=fontsize)

    ds = ax[0].plot(data['vel'], signal_limrad[iT, iR, :], linewidth=1.75, color='royalblue',
                    label='LIMRAD94 Doppler spectrum')
    ax[0].set_xlim(left=x_lim[0], right=x_lim[1])
    ax[0].set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax[0].set_ylabel('Doppler\nspectrum (dBZ)', fontweight='bold', fontsize=fontsize)
    ax[0].grid(linestyle=':')
    # ax2 = divider0.append_axes("bottom", size="50%", pad=0.08)

    dT = h.ts_to_dt(time[iT])
    # ax[0].text(0.01, 0.93, '{} UTC  at {:.2f} m ({})'.format(dT.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], rg, 'LIMRAD94'),
    #           horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)

    if 'vspec_norm' in kwargs:
        vhspec_norm = kwargs['vspec_norm']
        ax11 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
        nds = ax11.plot(data['vel'], vhspec_norm, linestyle='-', color='black',
                        label='normalized Doppler spectrum')
        ax11.set_xlim(left=x_lim[0], right=x_lim[1])
        ax11.set_ylim(bottom=0, top=1)
        ax11.set_ylabel('normalized\nspectrum (-)', fontweight='bold', fontsize=fontsize)
        # ax11.grid()

    if second_data_set:
        # find the closest spectra to the first device
        iT2 = h.argnearest(time2, time[iT])
        iR2 = h.argnearest(height2, rg)

        dT2 = h.ts_to_dt(time2[iT2])
        rg2 = height2[iR2]
        ax[0].step(vel2, signal_mira[iT2, iR2, :], color='darkred', label='MIRA Doppler Spectrum')

        ax[0].text(0.01, 0.85,
                   '{} UTC  at {:.2f} m ({})'.format(dT2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], rg2, 'MIRA'),
                   horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)

    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="2.5%", pad=0.05)
    cax0.axis('off')

    # fig.add_axes(cax0)

    # added these three lines
    if legend:
        lns = ds + nds if 'vspec_norm' in kwargs else ds
        labs = [l.get_label() for l in lns]
        ax[0].legend(lns, labs, loc='upper right')
        # plt.legend(loc='upper right')
    ia, ib = h.argnearest(data['vel'], x_lim[0]), h.argnearest(data['vel'], x_lim[1])
    img = ax[1].imshow(cwtmatr_spec[:, ia:ib], extent=[x_lim[0], x_lim[1], widths[-1], widths[0]],
                       cmap=colormap, aspect='auto', vmin=z_lim[0], vmax=z_lim[1])
    ax[1].set_ylabel('wavelet\nscale', fontweight='bold', fontsize=fontsize)
    ax[1].set_xlabel('Doppler Velocity (m/s)', fontweight='bold', fontsize=fontsize)
    ax[1].set_xlim(left=x_lim[0], right=x_lim[1])
    ax[1].set_ylim(bottom=widths[0], top=widths[-1])
    ax[1].set_yticks(np.linspace(widths[0], widths[-1], 4))
    # ax = plt.gca()
    ax[1].invert_yaxis()
    # Set the tick labels
    # ax[1].set_yticklabels([r'$2^{1.75}$', r'$2^{2.5}$', '$2^{3.25}$', '$2^{3.75}$'])
    ax[1].set_xticklabels([])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    fig.add_axes(cax)
    cbar = fig.colorbar(img, cax=cax, orientation="vertical")
    cbar.set_label('Magnitude\nof Similarity', fontsize=fontsize, fontweight='bold')
    ax[1].grid(linestyle=':')
    ax[1].xaxis.set_ticks_position('top')
    # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # plt.tight_layout(rect=[0, 0.01, 1, 1], h_pad=0.1)
    # plt.show()

    if 'features' in kwargs:
        vmin, vmax = [-0.5, len(VIS_Colormaps.categories['four_colors']) - 0.5]
        img = ax[2].pcolormesh(data['vel'], widths, features, cmap=VIS_Colormaps.four_colors_map, vmin=vmin, vmax=vmax)
        # ax[2].set_ylabel('wavelet\nscale', fontweight='bold', fontsize=fontsize)
        ax[2].set_xlabel('Doppler Velocity (m/s)', fontweight='bold', fontsize=fontsize)
        ax[2].set_xlim(left=x_lim[0], right=x_lim[1])
        ax[2].set_ylim(bottom=widths[0], top=widths[-1])
        ax[2].set_yticks(np.linspace(widths[0], widths[-1], 4))
        # ax = plt.gca()
        ax[2].invert_yaxis()
        # Set the tick labels
        ax[2].set_yticklabels([r'$2^{1.00}$', r'$2^{1.75}$', '$2^{2.50}$', '$2^{3.75}$'])
        ax[2].set_xticklabels([])
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        fig.add_axes(cax)
        cbar = fig.colorbar(img, cax=cax, orientation="vertical")

        categories = VIS_Colormaps.categories['four_colors']
        cbar.set_ticks(list(range(len(categories))))
        cbar.ax.set_yticklabels(categories)

        ax[2].grid(linestyle=':')
        ax[2].xaxis.set_ticks_position('top')

    return fig, ax


def container2DataArray(container):
    """convert the data_container to a xarray Dataset
    
    ``time`` dimension (named ``ts``) is converted to datatime.datetime and to datetime64

    Args:
        data (dict): data_container

    Returns:
        ``xarray.DataArray``
    """

    import xarray as xr

    dimlabel = container['dimlabel']
    var = container['var']

    # the dimlabel is not always named exactly as the key of the dimension
    #
    label2coord = {'time': 'dt', 'range': 'rg', 'vel': 'vel'}

    if 'time' in dimlabel:
        container['dt'] = [h.ts_to_dt(t) for t in container['ts']]
    coords = [container[label2coord[l]] for l in container['dimlabel']]

    name = container['system'] + ' ' + container['name']

    # strip off the actual arrays from the attrs
    attrs = {**container}
    attrs.pop('var', None)
    attrs.pop('mask', None)
    attrs.pop('dimlabel', None)
    [attrs.pop(label2coord[l], None) for l in container['dimlabel']]

    da = xr.DataArray(data=var,
                      dims=dimlabel,
                      name=name,
                      coords=coords,
                      attrs=attrs)
    return da


def roll_mean_2D(matrix, windowsize, dim):
    """
    Calculate a rolling mean over a given axis of a 2D (time x range) array
    Args:
        matrix (ndarray): 2D (time x range) matrix
        windowsize (int): size of the moving window
        dim (str): over which dimension to apply the mean, 'time' or 'range'

    Returns: 2D matrix of averaged values

    """
    dims = {dim: windowsize}
    da = xr.DataArray(matrix, dims=['time', 'range'])  # turn matrix into data array to use xarray rolling function
    da_roll = da.rolling(dims, center=True).mean(skipna=True)

    return da_roll.values


#########################################################################
###### new module


_DEFAULT_FIGSIZE = [14, 5.7]


def _new_fig(
        figsize: List[float] = None,
        figure: plt.figure = None,
        axis: plt.axis = None,
        **kwargs
) -> (plt.figure, plt.axis):
    """
    Opens or parses a matplotlib figure and axis.

    Args:
        figsize (optional): list of figure height and width
        figure (optional): already open matplotlib figure
        axis (optional): already open matplotlib axis

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **axis**: axis
    """
    if figsize is None:
        figsize = _DEFAULT_FIGSIZE

    if not (figure and axis):
        figure, axis = plt.subplots(1, figsize=figsize)
    return figure, axis


def _copy_data(
        data: dict or xr.Dataset,
        colormp: str = None,
        mask: np.array = None,
        rg_converter: bool = False,
        time_interval: List[datetime.datetime] = None,
        range_interval: List[float] = None,
        var_lims: List[float] = None,
        fontsize: int = 12,
        labelsize: int = 12,
        figsize: List[float] = None,
        fontweight: str = 'normal',
        cbar: bool = True,
        clim: List[float] = None,
        linewidth: int = 1.5,
        alpha: float = 0.95,
        **kwargs
) -> dict:
    """
        Copy data from an xarray or larda dict to the plot data structure 'pdata'.

    Args:
        data: xarray or larda dict
        colormp (optional): string key from matplotlib colormaps
        mask (optional): additional variable mask (invalid/missing data)
        rg_converter (optional): if True converting from m to km for 2D plots
        time_interval (optional): restricts the xaxis for time series and 2D data
        range_interval (optional): restricts the yaxis for 2D data
        var_lims (optional): new data limits, default: var_lims of data
        fontsize (optional): figure font size
        labelsize (optional): colorbar label size
        figsize (optional): figure size list of height and width
        fontweight (optional): axis label font weight
        cbar (optional): if False, hides the colorbar from the plot
        cmap (optional): string according to matplotlib colormap lists
        linewidth (optional): width of time series data line plot
        alpha (optional): alpha of time series data line plot

    Returns:
        pdata (plot data structure)

    """
    pdata = {}

    # check for availability of dimensions in larda container or xarray object
    # xarray
    if hasattr(data, 'coords'):
        if len(data.coords.dims) == 2:
            assert data.coords.dims == ('ts', 'rg'), f'attribute error, check coords ... wrong plot function for {data}'
            pdata['rg'] = data['rg'].values.copy()
            pdata['rg_unit'] = data.attrs['rg_unit']
            pdata['colormap_name'] = data.colormap if hasattr(data, 'colormap') else colormp
            pdata['cmap'] = pdata['colormap_name']
        else:
            assert data.coords.dims == ('time',), f'wrong plot function for {data["dimlabel"]}'

        try:
            pdata['ts'] = data['ts'].values.copy().astype('datetime64[s]').astype(np.int64)
        except:
            pdata['ts'] = data['time'].values.copy().astype('datetime64[s]').astype(np.int64)
        pdata['dimlabel'] = ['time', 'range'] if list(data.coords.dims) == ['ts', 'rg'] else list(data.coords.dims)
        pdata['mask'] = data.mask.values if 'mask' == None else mask
        pdata['var'] = data.values.copy()
        pdata['name'] = data.name
        pdata['var_unit'] = data.attrs['var_unit']
        pdata['system'] = data.system
        if var_lims is None:
            if 'var_lims' in data.attrs.keys():
                pdata['var_lims'] = data.attrs['var_lims']
            else:
                raise ValueError('No var_lims were provided!')

                # larda container
    else:
        if len(data['dimlabel']) == 2:
            assert data['dimlabel'] == ['time', 'range'], f'wrong plot function for {data["dimlabel"]}'
            pdata['rg'] = data['rg'].copy()
            pdata['rg_unit'] = data['rg_unit']
            pdata['colormap_name'] = data['colormap']
            pdata['cmap'] = pdata['colormap_name']
        else:
            assert data['dimlabel'] == ['time'], f'wrong plot function for {data["dimlabel"]}'

        pdata['dimlabel'] = data['dimlabel']
        pdata['mask'] = data['mask'].copy()
        pdata['var'] = data['var'].copy()
        pdata['name'] = data['name']
        pdata['var_unit'] = data['var_unit']
        pdata['system'] = data["system"]
        pdata['ts'] = data['ts'].copy()
        if var_lims is None:
            if 'var_lims' in data.keys():
                pdata['var_lims'] = data['var_lims']
            else:
                raise ValueError('No var_lims were provided!')

    pdata['dt'] = [datetime.datetime.utcfromtimestamp(time) for time in pdata['ts']]
    pdata['var'] = np.ma.masked_where(pdata['mask'], pdata['var'])
    pdata['time_interval'] = [pdata['dt'][0], pdata['dt'][-1]] if time_interval is None else time_interval

    if rg_converter:
        pdata['rg'] = data['rg'].copy() / 1000.0
        pdata['rg_unit'] = 'km'

    if 'range' in pdata['dimlabel']:
        pdata['range_interval'] = [pdata['rg'][0], pdata['rg'][-1]] if range_interval is None else range_interval

    pdata['fontsize'] = fontsize
    pdata['figsize'] = _DEFAULT_FIGSIZE if figsize is None else figsize
    pdata['labelsize'] = labelsize
    pdata['fontweight'] = fontweight
    pdata['cbar'] = cbar
    pdata['clim'] = clim
    pdata['linewidth'] = linewidth
    pdata['alpha'] = alpha
    return pdata


def _masked_jumps(
        pdata: dict,
        tdel_jumps: float = 60,
        tres: float = 5.0,
        **kwargs
) -> (list, np.ma.array):
    """
    Fills data gaps with fill_values and masks them.

    Args:
        pdata: plot data structure
        tdel_jumps (optional): time jump threshold in seconds
        tres (optional): target time resolution in seconds

    Returns:
        tuple with data corrected for gaps

        - **dt** (list): datetimes
        - **var** (np.ma.ndarray): var
    """

    # this is the last valid index
    dt = pdata['dt'].copy()
    var = pdata['var'].copy()
    var = var.astype(np.float32).filled(-999)
    jumps = np.where(np.diff(pdata['ts']) > tdel_jumps)[0]
    filled_profile = np.full(var.shape[1], -999) if len(var.shape) == 2 else -999
    for ind in jumps[::-1].tolist():
        logger.debug("masked jump at {} {}".format(ind, dt[ind - 1:ind + 2]))
        # and modify the dt_list
        dt.insert(ind + 1, dt[ind] + datetime.timedelta(seconds=tres))
        # add the fill array
        var = np.insert(var, ind + 1, filled_profile, axis=0)

    return dt, np.ma.masked_equal(var, -999)


def _get_cbar_limits(
        pdata: dict,
        is_class: bool = False,
        cbar_fraciton: float = 0.13,
        zlim: List[float] = None,
        **kwargs
) -> (float, float):
    """
    Returns the minimum and maximum value for the colorbar.

    Args:
        pdata:
        is_class (optional): if True increas colorbar portion for larger labels
        cbar_fraciton (optional): colorbar fraction
        zlim (optional): if provided, these colorbar limits will get passed through

    Returns:
        zlim - color bar axis limits
    """

    pdata['cbar_fraciton'] = cbar_fraciton
    if is_class:
        # make the figure a littlebit wider and
        # center the label dashes in the middle of the individual color box
        pdata['cbar_fraciton'] = 0.23
        return -0.5, len(VIS_Colormaps.categories[pdata['colormap_name']]) - 0.5

    if zlim is None:
        if 'var_lims' in pdata:
            return pdata['var_lims']
        else:
            return pdata['var'].min(), pdata['var'].max()
    else:
        return zlim


def _apply_2Dvar_converter(
        pdata: dict,
        var_converter: str = None,
        z_converter: str = None,
        **kwargs
) -> (np.ma.array, None or matplotlib.colors.LogNorm):
    """Converts a 2D numpy array for a given valid string.
    
    Args:
        pdata: 
        var_converter (optional): other name for variable converter string
        z_converter (optional): variable converter string

    Returns:
        tuple with

        - **var**: converted variable
        - **norm**: matplotlib log norm
    """
    var, norm = pdata['var'].copy(), None
    if var_converter is not None:
        z_converter = var_converter

    if z_converter is not None:
        if z_converter == 'log':
            norm = matplotlib.colors.LogNorm(vmin=pdata['vmin'], vmax=pdata['vmax'])
        else:
            var = h.get_converter_array(z_converter)[0](var)

    return var, norm


def _apply_1Dvar_converter(
        var: np.ma.array,
        var_converter: str = None,
        z_converter: str = None,
        **kwargs
) -> np.ma.array:
    """Converts a 1D numpy array for a given valid string.

    Args:
        var: 1D numpy array data
        var_converter (optional): other name for variable converter string
        z_converter (optional): variable converter string

    Returns:
        va - converted array

    """

    va = var.copy()
    if var_converter is not None:
        z_converter = var_converter

    if z_converter is not None:
        if z_converter == 'log':
            return va
        else:
            return h.get_converter_array(z_converter)[0](va)
    else:
        return va


def _apply_log_scaling(
        axis: plt.axis,
        var_converter: str = None,
        z_converter: str = None,
        **kwargs
) -> np.ma.array:
    """Converts the yaxis scale to log iff z_converter = 'log'

    Args:
        axis: matplotlib axis
        var_converter (optional): other name for variable converter string
        z_converter (optional): variable converter string

    Returns:
        va - converted array

    """

    if var_converter is not None:
        z_converter = var_converter
    if z_converter is not None:
        if z_converter == 'log':
            axis.set_yscale('log')
    return axis


def _get_colormap(
        pdata: dict,
) -> str or matplotlib.colors.ListedColormap:
    """Parse colormap string or check for additional color maps.

    Args:
        pdata: plot data

    Returns:
        cmap - colormap string
    """

    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if pdata['colormap_name'] in VIS_Colormaps.custom_colormaps.keys():
        return VIS_Colormaps.custom_colormaps[pdata['colormap_name']]
    else:
        return pdata['colormap_name']


def _add_contour(
        axis: plt.axis,
        contour: dict = None,
        rg_converter: bool = False,
        fontsize: int = 12,
        **kwargs
) -> plt.axis:
    """Plots contour lines with label ontop of an existing matplotlib axis.

    Args:
        axis: plot axis
        contour (optional): dictionary with keys 'data' and 'levels' (optional), where data is an xarray or larda dict
        rg_converter (optional): if True convert range from meter to kilometer
        fontsize (optional): size of contour labels

    Returns:
        tuple with

        - **ax**: axis
        - **pcont**:
    """
    pcont = None
    if contour is not None:
        cdata = contour['data']
        contour_color = kwargs['contour_col'] if 'contour_col' in kwargs else 'black'
        cstyle = {'linestyles': 'dashed', 'colors': contour_color, 'linewidths': 0.75}

        # check for larda container or xarray
        cdata_var = cdata['var'] if isinstance(cdata, dict) else cdata.values

        assert len(cdata) > 1, 'Contour data empty!'
        cdata_rg = np.divide(cdata['rg'], 1000.0) if rg_converter else cdata['rg']

        dt_c = [datetime.datetime.utcfromtimestamp(time) for time in cdata['ts']]
        if 'levels' in contour:
            pcont = axis.contour(dt_c, cdata_rg, cdata_var.T, contour['levels'], **cstyle)
        else:
            pcont = axis.contour(dt_c, cdata_rg, cdata_var.T, **cstyle)

        axis.clabel(pcont, fontsize=fontsize, inline=1, fmt='%1.1f°C', )
    return axis, pcont


def _format_timexaxis(
        axis: plt.axis,
        pdata: dict,
        **kwargs
) -> plt.axis:
    """Zoom to a specific time interval and format xaxis.

    Args:
        axis: plot axis
        pdata: plot structure data

    Returns:
        axis - plot axis
    """
    axis.set_xlim(pdata['time_interval'])
    axis.set_xlabel("Time [UTC]", fontsize=pdata['fontsize'], fontweight=pdata['fontweight'])
    return axis


def _set_xticks_and_xlabels(
        axis: plt.axis,
        time_extend: datetime.timedelta
) -> plt.axis:
    """This function sets the ticks and labels of the x-axis (only when the x-axis is time in UTC).

    Options:
        -   time_extend > 7 days:               major ticks every 2 day,  minor ticks every 12 hours
        -   7 days > time_extend > 2 days:      major ticks every day, minor ticks every  6 hours
        -   2 days > time_extend > 1 days:      major ticks every 12 hours, minor ticks every  3 hours
        -   1 days > time_extend > 6 hours:     major ticks every 3 hours, minor ticks every  30 minutes
        -   6 hours > time_extend > 1 hour:     major ticks every hour, minor ticks every  15 minutes
        -   else:                               major ticks every 15 minutes, minor ticks every  5 minutes

    Args:
        axis: axis in which the x-ticks and labels have to be set
        time_extend: time difference of t_end - t_start (format datetime.timedelta)

    Returns:
        axis - axis with new ticks and labels
    """

    if time_extend > datetime.timedelta(days=30):
        pass
    elif datetime.timedelta(days=30) > time_extend > datetime.timedelta(days=7):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        axis.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=range(1, 32, 2)))
        axis.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
    elif datetime.timedelta(days=7) > time_extend > datetime.timedelta(days=2):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        axis.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
        axis.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 6)))
    elif datetime.timedelta(days=2) > time_extend > datetime.timedelta(hours=25):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
        axis.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    elif datetime.timedelta(hours=25) > time_extend > datetime.timedelta(hours=6):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
        axis.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=6) > time_extend > datetime.timedelta(hours=2):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        axis.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
    elif datetime.timedelta(hours=2) > time_extend > datetime.timedelta(minutes=15):
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
        axis.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        axis.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))

    return axis


def _format_rangeyaxis(
        axis: plt.axis,
        pdata: dict,
        rg_converter: bool = False,
        **kwargs
) -> plt.axis:
    """Zoom to a specific range interval and format yaxis.

    Args:
        axis: plot axis
        pdata: plot structure data
        rg_converter: if True convert range from meter to kilometer

    Returns:
        axis - plot axis
    """
    if rg_converter:
        axis.set_ylim(pdata['range_interval'] / 1000.0)
        rg_unit = 'km'
    else:
        axis.set_ylim(pdata['range_interval'])
        rg_unit = pdata['rg_unit']

    ylabel = f"Height [{rg_unit}]"
    axis.set_ylabel(ylabel, fontsize=pdata['fontsize'], fontweight=pdata['fontweight'])
    return axis


def _format_xaxis(
        axis: plt.axis,
        pdata: dict,
        fontsize: int = 12,
        fontweight: str = 'normal',
        label: str = None,
        **kwargs
) -> plt.axis:
    """More general x axis formatting.

    Args:
        axis: plot axis
        pdata: plot data structure
        fontsize (optional): size of labels
        fontweight (optional): weight of labels
        label: label for xaxis

    Returns:
        axis - plot axis
    """

    axis.set_xlim(pdata['var_lims'])
    if label is None:
        label = f'{pdata["name"]} [{pdata["var_unit"]}]'
    axis.set_xlabel(label, fontsize=fontsize, fontweight=fontweight)
    return axis


def _format_yaxis(
        axis: plt.axis,
        pdata: dict,
        fontsize: int = 12,
        fontweight: str = 'normal',
        label: str = None,
        **kwargs
) -> plt.axis:
    """More general y axis formatting.

    Args:
        axis: plot axis
        pdata: plot data structure
        fontsize (optional): size of labels
        fontweight (optional): weight of labels
        label: label for xaxis

    Returns:
        axis - plot axis
    """
    if 'var_lims' in kwargs:
        axis.set_ylim(kwargs['var_lims'])
    else:
        axis.set_ylim(pdata['var_lims'])

    if label is None:
        label = f'{pdata["name"]} [{pdata["var_unit"]}]'
    axis.set_ylabel(label, fontsize=fontsize, fontweight=fontweight)
    return axis


def _plot_line(
        axis: plt.axis,
        pdata: dict,
        label_str: str,
        step: bool = False,
        **kwargs
) -> (plt.axis, matplotlib.lines.Line2D):
    """plot a line or step

    Args:
        axis: plot axis
        pdata: plot data structure
        label_str: label of line
        step (bool or str): 'pre', 'mid', 'post', default False

    Returns:
        tuple with

        - **ax**: plot axis
        - **line**: Line2D
    """
    if not step:
        line = axis.plot(
            matplotlib.dates.date2num(pdata['dt'][:]), pdata['var'][:],
            linewidth=pdata['linewidth'], alpha=pdata['alpha'], label=label_str
        )
    else:
        line = axis.step(
            matplotlib.dates.date2num(pdata['dt'][:]), pdata['var'][:],
            linewidth=pdata['linewidth'], alpha=pdata['alpha'], label=label_str,
            where=step
        )

    return axis, line



def _format_cbaraxis(
        figure: plt.figure,
        axis: plt.axis,
        pcmesh: plt.pcolormesh,
        pdata: dict,
        color_by: dict = None,
        is_scatter: bool = False,
        is_class: bool = False,
        remove: bool = False,
        clim: List[float] = None,
        **kwargs
) -> (plt.axis, plt.colorbar):
    """Formates the colorbar.

    Args:
        figure: plot figure
        axis: plot axis
        pcmesh: colormesh plot
        pdata: plot data structure
        color_by (optional): larda container
        is_scatter (optional): if True adapt colorbar for scatter plot
        is_class (optional): if True adapt colorbar with discrete labels
        remove (optional): if True remove colorbar
        clim (optional): colorbar limits

    Returns:
        tuple with

        - **ax**: plot axis
        - **cbar**: colorbar
    """

    if is_scatter:
        cmap = copy(plt.get_cmap(pdata['cmap']))

        if color_by is not None:
            z_string = f"median {color_by['name']} [{color_by['var_unit']}]"
            formstring = "%.2f"
        else:
            cmap.set_under('white', 1.0)
            z_string = "frequency of occurrence"
            if np.log10(pdata['var']).max() > 1000:
                formstring = '%.2E'
            else:
                formstring = '%.0f'

        cbar = figure.colorbar(pcmesh, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8, format=formstring)
        cbar.set_label(label=z_string, fontweight=pdata['fontweight'], fontsize=pdata['fontsize'])
        cbar.axis.tick_params(axis='both', which='major', labelsize=pdata['labelsize'], width=2, length=4)
        if clim is not None:
            cbar.mappable.set_clim(clim)
        cbar.aspect = 50

        return axis, cbar

    if not remove:

        cbar = figure.colorbar(pcmesh, ax=axis, fraction=pdata['cbar_fraciton'], pad=0.025)
        cbar.ax.set_ylabel(_axis_label_from_meta(pdata), fontweight=pdata['fontweight'], fontsize=pdata['fontsize'])
        cbar.ax.tick_params(axis='both', which='major', labelsize=pdata['labelsize'], width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

        if is_class:
            categories = VIS_Colormaps.categories[pdata['colormap_name']]
            cbar.set_ticks(list(range(len(categories))))
            cbar.ax.set_yticklabels(categories)
            cbar.ax.tick_params(labelsize=pdata['labelsize'])
            pdata['figsize'][0] -= 1.25  # change back to original

        return axis, cbar
    return axis, None


def _axis_label_from_meta(
        pdata: dict,
        **kwargs
) -> str:
    """Generates an automated axis label from data
    using system name, variable name and variable unit.

    Args:
        pdata: plot data structure

    Returns:
        str - axis label

    """
    return f"{pdata['system']} {pdata['name']} [{pdata['var_unit']}]"


def _format_axis(
        figure: plt.figure,
        axis: plt.axis,
        pcmesh: plt.pcolormesh,
        pdata: dict,
        pdatai: dict = None,
        color_by: dict = None,
        **kwargs
) -> (plt.axis, plt.colorbar):
    """Format all axis.

    Args:
        figure: plot figure
        axis: plot axis
        pcmesh: colormesh plot
        pdata: plot data structure
        pdatai (optional): additional plot data structure, formatting scatter plots

    Returns:
        tuple with

        - **ax**: axis
        - **cbar**: colorbar
        - **var_lims**: y axis limits
    """

    from matplotlib.ticker import AutoMinorLocator
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())

    if 'time' in pdata['dimlabel']:
        axis = _format_timexaxis(axis, pdata)
        logger.debug(f"time extend {pdata['time_interval']}")
        axis = _set_xticks_and_xlabels(axis, pdata['time_interval'][-1] - pdata['time_interval'][0])

    cbar = None
    is_class_or_status = 'class' in pdata['name'].lower() or 'status' in pdata['name'].lower()

    if pdata['dimlabel'] == ['time', 'range']:
        axis = _format_rangeyaxis(axis, pdata)
        logger.debug(f"range extend {pdata['range_interval']}")
        axis, cbar = _format_cbaraxis(figure, axis, pcmesh, pdata, is_class=is_class_or_status)

    elif pdata['dimlabel'] == ['time']:
        axis = _format_yaxis(axis, pdata, **kwargs)

    # second dataset for scatter plot yaxis
    if pdatai is not None:
        axis = _format_xaxis(axis, pdata, label=_axis_label_from_meta(pdata), fontsize=pdata['fontsize'])
        axis = _format_yaxis(axis, pdatai, label=_axis_label_from_meta(pdatai), fontsize=pdata['fontsize'])
        axis, cbar = _format_cbaraxis(figure, axis, pcmesh, pdata, is_scatter=True, clim=pdata['clim'], color_by=color_by)

    axis.tick_params(axis='both', which='both', right=True, top=True)
    axis.tick_params(axis='both', which='major', labelsize=pdata['labelsize'], width=3, length=5.5)
    axis.tick_params(axis='both', which='minor', width=2, length=3)

    return axis, cbar


def _set_title(
        axis: plt.axis,
        pdata: dict,
        title: str = None,
        **kwargs
) -> plt.axis:
    """Set/remove or auto generate a title.

    Args:
        axis: plot axis
        pdata: plot data structure
        title (optional): title string
        **title_loc (optional): where to position title ('center', 'left', 'right')

    Returns:
        axis - plot axis
    """
    loc = kwargs['title_loc'] if 'title_loc' in kwargs else 'center'
    if title is None:
        axis.set_title('')
        return axis

    if isinstance(title, str):
        axis.set_title(title, fontsize=pdata['fontsize'], loc=loc)

    if isinstance(title, bool) and title:
        # auto generated title
        title = f" {pdata['system']} -- {pdata['name']} -- {pdata['dt'][0]:%Y-%m-%d %H:%M} till {pdata['dt'][-1]:%Y-%m-%d %H:%M}"
        axis.set_title(title, fontsize=pdata['fontsize'], fontweight=pdata['fontweight'], loc=loc)
    return axis


def _get_line_label(
        pdata: dict,
        label: (str or bool) = '',
        **kwargs
) -> str:
    """Parse or auto generate a line label.

    Args:
        pdata: plot data structure
        label (optional): label name if string or autogenerated label if True, else no label

    Returns:
        string
    """
    if isinstance(label, bool) and label:
        return pdata['system'] + pdata['variable_name']
    elif isinstance(label, str) and len(label) > 0:
        return label
    else:
        return ''


def _Freedman_Diaconis(
        var: np.ma.array,
        name: str = 'unknown variable',
        Nbins: int = None,
        **kwargs
):
    """Parse or calculate the optimum number of bins for a scatter plot.

    Args:
        var: array
        name: variable name
        Nbins (optional): given number of bins

    Returns:
        Nbins - number of bins

    References:
        Freedman-Diaconis rule: h=2×IQR×n−1/3. number of bins is (max−min)/h, where n is the number of observations
        https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    """
    try:
        if Nbins is None:
            Nbins = int((np.nanmax(var) - np.nanmin(var)) // (2 * (np.nanquantile(var, 0.75) - np.nanquantile(var, 0.25)) * len(var) ** (-1 / 3)))
    except OverflowError:
        print(f'var {name}: len is {len(var)}, '
              f'IQR is {np.nanquantile(var, 0.75)} - {np.nanquantile(var, 0.25)},'
              f'max is {np.nanmax(var)}, min is {np.nanmin(var)}')
        Nbins = 100
    return Nbins


def _create_histogram(
        var1: np.ma.array,
        var2: np.ma.array,
        Nbins: int = 100,
        x_lim: List[float] = None,
        y_lim: List[float] = None,
        **kwargs
) -> dict:
    """Calculates a histogram from two given arrays with same size

    Args:
        var1: array xdata
        var2: array ydata
        Nbins (optional): number of bin in histogram
        x_lim (optional): x axis limit
        y_lim (optional): y axis limit

    Returns:
        hist - dictionary with histogram, x and y edges
    """
    assert var1.shape == var2.shape, RuntimeError('Provide two dataset sith same size!')

    if x_lim is None:
        x_lim = [np.nanmin(var1), np.nanmax(var1)]
    if y_lim is None:
        y_lim = [np.nanmin(var2), np.nanmax(var2)]

    # create histogram plot
    hist, xedges, yedges = np.histogram2d(var1, var2, bins=Nbins, range=[x_lim, y_lim])
    hist = np.ma.masked_less_equal(hist, 0)
    return {'H': hist, 'xedges': xedges, 'yedges': yedges}


def _color_by_3rd_variable(
        var1: np.ma.array,
        var2: np.ma.array,
        var3: np.ma.array,
        H: dict,
        Nbins: int = 100,
        **kwargs
) -> np.ma.array:
    """Recalculates a given histogram for a given 3rd variable.

    Args:
        var1: array xdata
        var2: array ydata
        var3: array zdata
        H: given histogram dict keys = H, xedges, yedges
        Nbins: number of bin in histogram

    Returns:
        hist - new histogram array
    """
    # overwrite H
    hist = np.zeros(H['H'].shape)
    # get the bins of the 2d histogram using digitize
    x_coords, y_coords = np.digitize(var1, H['xedges']), np.digitize(var2, H['yedges'])

    # find unique bin combinations = pixels in scatter plot
    # sort x and y coordinates using lexsort
    # lexsort sorts by multiple columns, first by y_coords then by x_coords
    newer_order = np.lexsort((x_coords, y_coords))
    x_coords, y_coords, var3 = x_coords[newer_order], y_coords[newer_order], var3[newer_order]
    first_hit_y = np.searchsorted(y_coords, np.arange(1, Nbins + 2))
    first_hit_y.sort()
    first_hit_x = [np.searchsorted(x_coords[first_hit_y[j]:first_hit_y[j + 1]], np.arange(1, Nbins + 2))
                   + first_hit_y[j] for j in np.arange(Nbins)]

    for x in range(Nbins):
        for y in range(Nbins):
            hist[y, x] = np.nanmedian(var3[first_hit_x[x][y]: first_hit_x[x][y + 1]])

    return hist


def _get_pcmesh_kwargs(
        H: np.ma.array,
        clim: int = None,
        scale: str = None,
        cmap: str = 'viridis',
        **kwargs
) -> dict:
    """Returns a dict with vmin, vmax and norm for pcolormesh plot.

    Args:
        H: histogram
        clim: given limits
        scale: scaling strategy
        cmap: colormap

    Returns:
        dict - pcmesh
    """
    if clim is None:
        limits = [1, round(np.max(H))]
    else:
        limits = clim

    pcmesh_kwargs = {}
    if scale == 'lin':
        pcmesh_kwargs.update({'vmin': limits[0], 'vmax': limits[1]})
    else:
        pcmesh_kwargs.update({'vmin': max(0.1, limits[0]), 'vmax': min(1.0e12, limits[1])})
        pcmesh_kwargs.update({'norm': matplotlib.colors.LogNorm(**pcmesh_kwargs)})

    pcmesh_kwargs['cmap'] = cmap

    return pcmesh_kwargs


def _add_regression_info(
        axis: plt.axis,
        pdata1: dict,
        pdata2: dict,
        info: bool = False,
        **kwargs
) -> plt.axis:
    """Add textbox with y axis intersetion, slope and r^2.

    Args:
        axis: plot axis
        pdata1: plot data structure
        pdata2: plot data structure
        info:

    Returns:
        axis - plot axis
    """
    if info:
        s, i, r, p, std_err = stats.linregress(pdata1['var'], pdata2['var'])
        axis.text(0.01, 0.93,
                f'slope = {s:5.3f}\nintercept = {i:5.3f}\nR^2 = {r * r:5.3f}',
                horizontalalignment='left',
                verticalalignment='center',
                transform=axis.transAxes,
                fontweight=pdata1['fontweight'],
                labelsize=pdata1['fontsize'])
    return axis


def plot_timeheight2(
        data: dict or xr.Dataset,
        **kwargs: dict
) -> (plt.figure, plt.axis):
    """Plot a timeheight larda container or xarray Dataset

    Args:
        data: data container
        **figure (optional): already open matplotlib figure
        **axis (optional): already open matplotlib axis
        **colormp (optional): string key from matplotlib colormaps
        **mask (optional): additional variable mask (invalid/missing data)
        **rg_converter (optional): if True converting from m to km for 2D plots
        **time_interval (optional): restricts the xaxis for time series and 2D data
        **range_interval (optional): restricts the yaxis for 2D data
        **var_lims (optional): new data limits, default: var_lims of data
        **fontsize (optional): figure font size
        **labelsize (optional): colorbar label size
        **fontweight (optional): axis label font weight
        **cbar (optional): if False, hides the colorbar from the plot
        **linewidth (optional): width of time series data line plot
        **alpha (optional): alpha of time series data line plot
        **zlim (optional): if provided, these colorbar limits will get passed through
        **var_converter (optional): other name for variable converter string
        **z_converter (optional): variable converter string
        **contour (optional): dictionary with keys 'data' and 'levels' (optional), where data is an xarray or larda dict
        **contour_col (optional): string defining the color of the contour, default is 'black'
        **label (optional): label name if string or autogenerated label if True, else no label
        **clim (optional): colorbar limits
        **title (optional): title string
        **title_loc (optional): location string for title ('center', 'left', 'right')

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    pdata = _copy_data(data, **kwargs)
    pdata['dt'], pdata['var'] = _masked_jumps(pdata, **kwargs)

    # add space for class or detection status labels
    is_classification = 'class' in pdata['name'].lower() or 'status' in pdata['name'].lower()
    if is_classification:
        pdata['figsize'][0] += 1.25

    pdata['vmin'], pdata['vmax'] = _get_cbar_limits(pdata, is_class=is_classification, **kwargs)
    pdata['var'], pdata['norm'] = _apply_2Dvar_converter(pdata, **kwargs)
    cmap_labels = _get_colormap(pdata)

    if pdata['norm'] is None:
        vlims = {'vmin': pdata['vmin'], 'vmax': pdata['vmax']}
    else:
        vlims = {'norm': pdata['norm']}

    figsize = kwargs['figsize'] if 'figsize' in kwargs else pdata['figsize']
    kwargs.pop('figsize') if 'figsize' in kwargs else None
    figure, axis = _new_fig(figsize=figsize, **kwargs)
    # pop figure and axis keywords after first use
    kwargs.pop('axis') if 'axis' in kwargs else None
    kwargs.pop('figure') if 'figure' in kwargs else None

    pcmesh = axis.pcolorfast(
        matplotlib.dates.date2num(pdata['dt']),
        pdata['rg'],
        pdata['var'][:-1, :-1].T,
        cmap=cmap_labels,
        **vlims
    )

    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs else pdata['fontsize']
    kwargs.pop('fontsize') if 'fontsize' in kwargs else None
    axis, cont = _add_contour(axis, fontsize=fontsize, **kwargs)
    axis, cbar = _format_axis(figure, axis, pcmesh, pdata, is_class=is_classification, **kwargs)
    axis = _set_title(axis, pdata, **kwargs)

    plt.subplots_adjust(right=0.99)
    figure.tight_layout()
    return figure, axis


def plot_timeseries2(
        data: dict or xr.Dataset,
        **kwargs: dict
) -> (plt.figure, plt.axis):
    """Plot a timeseries data container or xarray dataset.

    Args:
        data: data container
        **figure (optional): already open matplotlib figure
        **axis (optional): already open matplotlib axis
        **colormp (optional): string key from matplotlib colormaps
        **mask (optional): additional variable mask (invalid/missing data)
        **time_interval (optional): restricts the xaxis for time series and 2D data
        **var_lims (optional): new data limits, default: var_lims of data
        **figsize (optional): size of the figure in inches
        **fontsize (optional): figure font size
        **labelsize (optional): colorbar label size
        **fontweight (optional): axis label font weight
        **cbar (optional): if False, hides the colorbar from the plot
        **linewidth (optional): width of time series data line plot
        **alpha (optional): alpha of time series data line plot
        **zlim (optional): if provided, these colorbar limits will get passed through
        **var_converter (optional): other name for variable converter string
        **z_converter (optional): variable converter string
        **contour (optional): dictionary with keys 'data' and 'levels' (optional), where data is an xarray or larda dict
        **label (optional): label name if string or autogenerated label if True, else no label
        **clim (optional): colorbar limits
        **title (optional): title string
        **title_loc (optional): location string for title ('center', 'left', 'right')
        **x_lim (optional): x axis limit
        **y_lim (optional): y axis limit


    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    pdata = _copy_data(data, **kwargs)
    pdata['dt'], pdata['var'] = _masked_jumps(pdata, **kwargs)

    pdata['var'] = _apply_1Dvar_converter(pdata['var'], **kwargs)
    label_str = _get_line_label(data, **kwargs)

    figsize = kwargs['figsize'] if 'figsize' in kwargs else pdata['figsize']
    kwargs.pop('figsize') if 'figsize' in kwargs else None
    figure, axis = _new_fig(figsize=figsize, **kwargs)

    # pop figure and axis keywords after first use
    kwargs.pop('axis') if 'axis' in kwargs else None
    kwargs.pop('figure') if 'figure' in kwargs else None

    axis, line = _plot_line(axis, pdata, label_str, **kwargs)

    axis = _apply_log_scaling(axis, **kwargs)
    axis, _ = _format_axis(figure, axis, line, pdata, **kwargs)
    axis = _set_title(axis, pdata, **kwargs)

    plt.subplots_adjust(right=0.99)
    figure.tight_layout()

    return figure, axis


def plot_scatter2(
        data1: dict or xr.Dataset,
        data2: dict or xr.Dataset,
        identity_line: bool = False,
        figsize: List[float] = None,
        color_by: dict = None,
        scale: str = 'lin',
        Nbins: int = None,
        **kwargs
) -> (plt.figure, plt.axis):
    """Scatter plot for variable comparison between two devices or variables.

    Args:
        data1: container 1st device
        data2: container 2nd device
        identity_line (optional): plot 1:1 line if True
        figsize (optional): size of the figure in inches
        color_by (optional): data container 3rd device
        scale (optional): 'lin' or 'log' --> if you get a ValueError from matplotlib.colors
                          try setting scale to lin, log does not work for negative values!
        Nbins (optional): number of bins for histogram
        **figure (optional): already open matplotlib figure
        **axis (optional): already open matplotlib axis
        **colormp (optional): string key from matplotlib colormaps
        **mask (optional): additional variable mask (invalid/missing data)
        **var_lims (optional): new data limits, default: var_lims of data
        **fontsize (optional): figure font size
        **labelsize (optional): colorbar label size
        **fontweight (optional): axis label font weight
        **cbar (optional): if False, hides the colorbar from the plot
        **linewidth (optional): width of time series data line plot
        **alpha (optional): alpha of time series data line plot
        **zlim (optional): if provided, these colorbar limits will get passed through
        **var_converter (optional): other name for variable converter string
        **z_converter (optional): variable converter string
        **contour (optional): dictionary with keys 'data' and 'levels' (optional), where data is an xarray or larda dict
        **label (optional): label name if string or autogenerated label if True, else no label
        **clim (optional): colorbar limits
        **title (optional): title string
        **title_loc (optional): location string for title ('center', 'left', 'right')
        **x_lim (optional): x axis limit
        **y_lim (optional): y axis limit

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    def _compare_grids(data1, data2, var):
        return (data1[var] == data2[var]).all()

    same_grid = _compare_grids(data1, data2, 'ts')
    same_grid *= _compare_grids(data1, data2, 'rg')

    if not same_grid:
        raise RuntimeError('Provided dataset not on same grid!')

    pdata1 = _copy_data(data1)
    pdata2 = _copy_data(data2)
    pdata1['dimlabel'] = ['var_name']

    combined_mask = pdata1['mask'] + pdata2['mask']
    pdata1['var'] = _apply_1Dvar_converter(pdata1['var'][~combined_mask].ravel(), **kwargs)
    pdata2['var'] = _apply_1Dvar_converter(pdata2['var'][~combined_mask].ravel(), **kwargs)

    Nbins = _Freedman_Diaconis(pdata1['var'], name=pdata1['name'], Nbins=Nbins, **kwargs)
    hist = _create_histogram(pdata1['var'], pdata2['var'], Nbins=Nbins, **kwargs)

    if color_by is not None:
        print("Coloring scatter plot by {}...\n".format(color_by['name']))
        var3 = color_by['var'][~combined_mask].ravel()
        pdata1['clim'] = color_by['var_lims']
        hist['H'] = _color_by_3rd_variable(pdata1['var'], pdata2['var'], var3, hist, Nbins)

    X, Y = np.meshgrid(hist['xedges'], hist['yedges'])
    pcmesh_kwargs = _get_pcmesh_kwargs(hist['H'], clim=pdata1['clim'], scale=scale, cmap=pdata1['cmap'])
    pdata1['clim'] = [pcmesh_kwargs['vmin'], pcmesh_kwargs['vmax']]

    figsize = np.repeat(min(figsize), 2) if figsize is not None else [6, 6]
    if pdata1['cbar']:
        figsize[0] += 2

    figure, axis = _new_fig(figsize=figsize, **kwargs)
    # pop figure and axis keywords after first use
    kwargs.pop('axis') if 'axis' in kwargs else None
    kwargs.pop('figure') if 'figure' in kwargs else None

    pcmesh = axis.pcolormesh(X, Y, np.transpose(hist['H']), **pcmesh_kwargs)
    axis = _add_regression_info(axis, pdata1['var'], pdata2['var'], **kwargs)

    # helper lines (1:1), ...
    if identity_line:
        axis = _add_identity(axis, color='salmon', ls='-')
    axis = _apply_log_scaling(axis, **kwargs)

    axis, cbar = _format_axis(figure, axis, pcmesh, pdata1, pdata2, color_by=color_by)
    axis = _set_title(axis, pdata1, **kwargs)

    plt.grid(b=True, which='both', color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    return figure, axis

#########################

