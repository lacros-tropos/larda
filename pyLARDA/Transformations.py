#!/usr/bin/python3


import datetime
import sys

import matplotlib
import numpy as np
from copy import copy

# import itertools

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# scientific python imports
import scipy.interpolate
from scipy import stats

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
    assert datadict1['dimlabel'] == datadict2['dimlabel']
    new_data['dimlabel'] = datadict1['dimlabel']
    container_type = datadict1['dimlabel']

    if container_type == ['time', 'range']:
        logger.debug("{} {} {}".format(datadict1['ts'].shape, datadict1['rg'].shape, datadict1['var'].shape))
        logger.debug("{} {} {}".format(datadict2['ts'].shape, datadict2['rg'].shape, datadict2['var'].shape))
    thisjoint = datadict1['ts'].shape[0]
    new_data["joints"] = datadict1.get('joints', []) + [thisjoint] + datadict2.get('joints', [])
    logger.debug("joints {}".format(new_data['joints']))
    new_data['filename'] = h.flatten([datadict1['filename']] + [datadict2['filename']])

    assert datadict1['paraminfo'] == datadict2['paraminfo']
    new_data['paraminfo'] = datadict1['paraminfo']
    if container_type == ['time', 'range']:
        assert datadict1['rg_unit'] == datadict2['rg_unit']
        new_data['rg_unit'] = datadict1['rg_unit']
        assert datadict1['colormap'] == datadict2['colormap']
        new_data['colormap'] = datadict1['colormap']
        assert np.all(datadict1['rg'] == datadict2['rg']), (datadict1['rg'], datadict2['rg'])
    if 'vel' in container_type:
        assert np.all(datadict1['vel'] == datadict2['vel']), "vel coordinate arrays not equal"
        new_data['vel'] = datadict1['vel']
    assert datadict1['var_unit'] == datadict2['var_unit']
    new_data['var_unit'] = datadict1['var_unit']
    assert datadict1['var_lims'] == datadict2['var_lims']
    new_data['var_lims'] = datadict1['var_lims']
    assert datadict1['system'] == datadict2['system']
    new_data['system'] = datadict1['system']
    assert datadict1['name'] == datadict2['name']
    new_data['name'] = datadict1['name']
    logger.debug(new_data['dimlabel'])
    logger.debug(new_data['paraminfo'])

    if container_type == ['time', 'range'] \
            or container_type == ['time', 'range', 'vel']:
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


def interpolate2d(data, mask_thres=0.1, **kwargs):
    """interpolate timeheight data container

    Args:
        mask_thres (float, optional): threshold for the interpolated mask 
        **new_time (np.array): new time axis
        **new_range (np.array): new range axis
    """

    var = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    logger.debug('var min {}'.format(data['var'][~data['mask']].min()))

    kx, ky = 1, 1
    interp_var = scipy.interpolate.RectBivariateSpline(
        data['ts'], data['rg'], var,
        kx=kx, ky=ky)
    interp_mask = scipy.interpolate.RectBivariateSpline(
        data['ts'], data['rg'], data['mask'].astype(np.float),
        kx=kx, ky=ky)

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']
    new_var = interp_var(new_time, new_range, grid=True)
    new_mask = interp_mask(new_time, new_range, grid=True)

    # print('new_mask', new_mask)
    new_mask[new_mask > mask_thres] = 1
    new_mask[new_mask < mask_thres] = 0
    # print('new_mask', new_mask)

    # print(new_var.shape, new_var)
    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = new_var
    interp_data['mask'] = new_mask
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
    """

    if type(datalist) == list and len(datalist) > 1:
        assert np.all(datalist[0]['rg'] == datalist[1]['rg'])
        assert np.all(datalist[0]['ts'] == datalist[1]['ts'])

    # use the first dict as the base
    new_data = {**datalist[0]} if type(datalist) == list else {**datalist}
    new_data.update(keys_to_update)

    new_data['var'], new_data['mask'] = func(datalist)
    if type(datalist) == list:
        new_data['history'] = {
            'filename': [e['filename'] for e in datalist],
            'paraminfo': [e['paraminfo'] for e in datalist],
        }
    else:
        new_data['history'] = {'filename': datalist['filename'],
                               'paraminfo': datalist['paraminfo']}

    return new_data


def slice_container(data, value={}, index={}):
    """slice a data_container either by values or indices (or combination of both)
    
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

    """
    dim_to_coord_array = {'time': 'ts', 'range': 'rg', 'vel': 'vel'}
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
            else:
                sliced_data[coord_name] = sliced_data[coord_name][0]
    logger.debug("new_dimlabel {}".format(new_dimlabel))
    sliced_data['dimlabel'] = new_dimlabel
    # actual slicing the variable
    slicer = tuple([slicer_dict[dim] for dim in data['dimlabel']])
    sliced_data['var'] = data['var'][slicer]
    sliced_data['mask'] = data['mask'][slicer]
    logger.info('sliced {} to shape {}'.format(data['var'].shape, sliced_data['var'].shape))
    return sliced_data


def plot(data):
    """call correct function based on type"""


def plot_timeseries(data, **kwargs):
    """plot a timeseries data container

    Args:
        data (dict): data container
        **time_interval (list dt): constrain plot to this dt
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
    """
    assert data['dimlabel'] == ['time'], 'wrong plot function for {}'.format(data['dimlabel'])
    time_list = data['ts']
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    # this is the last valid index
    var = var.filled(-999)
    jumps = np.where(np.diff(time_list) > 60)[0]
    for ind in jumps[::-1].tolist():
        logger.debug("jump at {} {}".format(ind, dt_list[ind - 1:ind + 2]))
        # and modify the dt_list
        dt_list.insert(ind + 1, dt_list[ind] + datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind + 1, -999, axis=0)

    var = np.ma.masked_equal(var, -999)

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    vmin, vmax = data['var_lims']
    logger.debug("varlims {} {}".format(vmin, vmax))
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            # plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            ax.set_yscale('log')
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)

    ax.plot(dt_list, var)

    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    ax.set_ylim([vmin, vmax])

    # ax.set_ylim([height_list[0], height_list[-1]])
    # ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    # ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)

    ylabel = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    time_extend = dt_list[-1] - dt_list[0]
    logger.debug("time extend {}".format(time_extend))
    if time_extend > datetime.timedelta(hours=6):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 30]))
    elif time_extend > datetime.timedelta(hours=1):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
    else:
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.xaxis.set_minor_locator(
            matplotlib.dates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14,
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)

    return fig, ax


def plot_timeheight(data, **kwargs):
    """plot a timeheight data container

    Args:
        data (dict): data container
        **time_interval (list dt): constrain plot to this dt
        **range_interval (list float): constrain plot to this ranges
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
        **contour: add a countour
        **fig_size (list): size of figure, default is 10, 5.7
    """
    assert data['dimlabel'] == ['time', 'range'], 'wrong plot function for {}'.format(data['dimlabel'])
    time_list = data['ts']
    range_list = data['rg']
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    # this is the last valid index
    var = var.astype(np.float64).filled(-999)
    jumps = np.where(np.diff(time_list) > 60)[0]
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
    if data['name'] in ['CLASS']:
        vmin, vmax = [-0.5, 10.5]
        # make the figure a littlebit wider and 
        # use more space for the colorbar
        fig_size[0] = fig_size[0] + 2.5
        fraction_color_bar = 0.23
        assert data['colormap'] == 'cloudnet_target'


    else:
        vmin, vmax = data['var_lims']
    logger.debug("varlims {} {}".format(vmin, vmax))
    plotkwargs = {}
    if 'z_converter' in kwargs:
        if kwargs['z_converter'] == 'log':
            plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    colormap = data['colormap']
    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if colormap in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap]

    fig, ax = plt.subplots(1, figsize=fig_size)
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]),
                           range_list[:],
                           np.transpose(var[:, :]),
                           # cmap=VIS_Colormaps.carbonne_map,
                           cmap=colormap,
                           vmin=vmin, vmax=vmax,
                           **plotkwargs
                           )

    if 'contour' in kwargs:
        cdata = kwargs['contour']['data']

        dt_c = [datetime.datetime.utcfromtimestamp(time) for time in cdata['ts']]
        if 'levels' in kwargs['contour']:
            cont = ax.contour(dt_c, cdata['rg'],
                              np.transpose(cdata['var']),
                              kwargs['contour']['levels'],
                              linestyles='dashed', colors='black')
        else:
            cont = ax.contour(dt_c, cdata['rg'],
                              np.transpose(cdata['var']),
                              linestyles='dashed', colors='black')
        ax.clabel(cont, fontsize=10, inline=1, fmt='%1.1f', )

    cbar = fig.colorbar(pcmesh, fraction=fraction_color_bar, pad=0.025)
    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    if 'range_interval' in kwargs.keys():
        ax.set_ylim(kwargs['range_interval'])

    # ax.set_ylim([height_list[0], height_list[-1]])
    # ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    # ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ylabel = 'Height [{}]'.format(data['rg_unit'])
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    if data['var_unit'] == "":
        z_string = "{} {}".format(data["system"], data["name"])
    else:
        z_string = "{} {} [{}]".format(data["system"], data["name"], data['var_unit'])
    cbar.ax.set_ylabel(z_string, fontweight='semibold', fontsize=15)
    if data['name'] in ['CLASS']:
        categories = VIS_Colormaps.categories['cloudnet_target']
        cbar.set_ticks(list(range(len(categories))))
        cbar.ax.set_yticklabels(categories)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    time_extend = dt_list[-1] - dt_list[0]
    logger.debug("time extend {}".format(time_extend))
    if time_extend > datetime.timedelta(hours=6):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 30]))
    elif time_extend > datetime.timedelta(hours=1):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
    else:
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.xaxis.set_minor_locator(
            matplotlib.dates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]))

    # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14,
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)
    cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
    if data['name'] in ['CLASS']:
        cbar.ax.tick_params(labelsize=11)

    ax.set_title('time-height-plot ' + data['system'] + ' ' + data['name'], size=20)

    plt.subplots_adjust(right=0.99)
    return fig, ax


def plot_scatter(data_container1, data_container2, identity_line=True, **kwargs):
    """scatter plot for variable comparison between two devices or variables

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        **identity_line (bool): plot 1:1 line if True
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **custom_offset_lines (float): plot 4 extra lines for given distance
    """
    var1_tmp = data_container1
    var2_tmp = data_container2

    combined_mask = np.logical_or(var1_tmp['mask'], var2_tmp['mask'])

    # convert var from linear unit with any converter given in helpers
    if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
        var1 = h.get_converter_array(kwargs['z_converter'])[0](var1_tmp['var'][~combined_mask].ravel())
        var2 = h.get_converter_array(kwargs['z_converter'])[0](var2_tmp['var'][~combined_mask].ravel())
    else:
        var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
        var2 = var2_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [var1.min(), var1.max()]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [var2.min(), var2.max()]

    # create histogram plot
    s, i, r, p, std_err = stats.linregress(var1, var2)
    H, xedges, yedges = np.histogram2d(var1, var2, bins=120, range=[x_lim, y_lim])

    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, figsize=(6, 6))

    ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm())

    ax.text(0.01, 0.93, 'slope = {:5.3f}\nintercept = {:5.3f}\nR^2 = {:5.3f}'.format(s, i, r ** 2),
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # helper lines (1:1), ...
    if identity_line: add_identity(ax, color='salmon', ls='-')

    if 'custom_offset_lines' in kwargs:
        offset = np.array([kwargs['custom_offset_lines'], kwargs['custom_offset_lines']])
        for i in [-2, -1, 1, 2]: ax.plot(x_lim, x_lim + i * offset, color='salmon', linewidth=0.7, linestyle='--')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('{} {} ({})'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']))
    ax.set_ylabel('{} {} ({})'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']))
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='both', which='both', right=True, top=True)

    return fig, ax


def plot_frequency_of_ocurrence(data, legend=True, **kwargs):
    """scatter plot for variable comparison between two devices

    Args:
        data (dict): container of Ze values
        **n_bins (integer): number of bins for reflectivity values (x-axis)
        **x_lim (list): limits of x-axis, default: data['var_lims']
        **y_lim (list): limits of y-axis, default: minimum and maximum of data['rg']
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **custom_offset_lines (float): plot 4 extra lines for given distance
        **sensitivity_limit (np.array): 1-Dim array containing the minimum sensitivity values for each range
    """

    n_ts = data['ts'].size
    n_rg = data['rg'].size

    # create a mask for fill_value = -999. because numpy.histogram can't handle masked values properly
    masked_vals = np.ones((n_ts, n_rg))
    masked_vals[data['var'] == -999.0] = 0.0
    var = np.ma.masked_less_equal(data['var'], -999.0)

    # check for key word arguments
    if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
        var = h.get_converter_array(kwargs['z_converter'])[0](var)
        if kwargs['z_converter'] == 'lin2z': data['var_unit'] = 'dBZ'

    n_bins = kwargs['n_bins'] if 'n_bins' in kwargs else 100
    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else data['var_lims']
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [data['rg'].min(), data['rg'].max()]

    # create bins of x and y axsis
    x_bins = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins = data['rg']

    # initialize array
    H = np.zeros((n_bins - 1, n_rg))

    for irg in range(n_rg):
        H[:, irg] = np.histogram(var[:, irg], bins=x_bins, density=False, weights=masked_vals[:, irg])[0]

    H = np.ma.masked_equal(H, 0.0)

    # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
    cmap = copy(plt.get_cmap('viridis'))
    cmap.set_under('white', 1.0)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    pcol = ax.pcolormesh(x_bins, y_bins, H.T, vmin=0.01, vmax=20, cmap=cmap, label='histogram')

    cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8)
    cbar.set_label(label="Frequencies of occurrence of {} values ".format(data['name']), fontweight='bold')
    cbar.aspect = 80

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('{} {} ({})'.format(data['system'], data['name'], data['var_unit']), fontweight='bold')
    ax.set_ylabel('Height ({})'.format(data['rg_unit']), fontweight='bold')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)

    if 'sensitivity_limit' in kwargs:
        sens_lim = kwargs['sensitivity_limit']
        if 'z_converter' in kwargs and kwargs['z_converter'] != 'log':
            sens_lim = h.get_converter_array(kwargs['z_converter'])[0](sens_lim)

        ax.plot(sens_lim, y_bins, linewidth=2.0, color='red', label='sensitivity limit')

    if 'range_offset' in kwargs:
        rg = kwargs['range_offset']
        ax.plot(x_lim, [rg[0]]*2, linestyle='-.', linewidth=1, color='black', alpha=0.5, label='chirp shift')
        ax.plot(x_lim, [rg[1]]*2, linestyle='-.', linewidth=1, color='black', alpha=0.5)

    plt.grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
    if legend: plt.legend(loc='upper left')
    fig.tight_layout()

    return fig, ax


def add_identity(axes, *line_args, **line_kwargs):
    """helper function for the scatter plot"""
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
            *data2 (dict or numpy.ndarray):     1.  data container of a second device, or
                                                2.  numpy array dimensions (time, height, 2) containing
                                                    noise threshold and mean noise level for each spectra
                                                    in linear units [mm6/m3]
            **z_converter (string): convert var before plotting use eg 'lin2z'
            **velmin (float): minimum x axis value
            **velmax (float): maximum x axis value
            **vmin (float): minimum y axis value
            **vmax (float): maximum y axis value
            **save (string): location where to save the pngs
            **fig_size (list): size of png, default is [10, 5.7]

        Returns:  
            tuple with

            - fig (pyplot figure): contains the figure of the plot 
              (for multiple spectra, the last fig is returned)
            - ax (pyplot axis): contains the axis of the plot 
              (for multiple spectra, the last ax is returned)
        """

    fsz = 13
    velocity_min = -8.0
    velocity_max = 8.0

    n_time, n_height = data['ts'].size, data['rg'].size
    vel = data['vel'].copy()

    time, height, var = h.reshape_spectra(data)

    velmin = kwargs['velmin'] if 'velmin' in kwargs else max(min(vel), velocity_min)
    velmax = kwargs['velmax'] if 'velmax' in kwargs else min(max(vel), velocity_max)

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]

    vmin = kwargs['vmin'] if 'vmin' in kwargs else data['var_lims'][0]
    vmax = kwargs['vmax'] if 'vmax' in kwargs else data['var_lims'][1]

    logger.debug("x-axis varlims {} {}".format(velmin, velmax))
    logger.debug("y-axis varlims {} {}".format(vmin, vmax))
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
        var = h.get_converter_array(kwargs['z_converter'])[0](var)

    name = kwargs['save'] if 'save' in kwargs else ''

    if len(args) > 0:
        if type(args[0]) == dict:
            data2 = args[0]
            vel2 = data2['vel'].copy()
            time2, height2, var2 = h.reshape_spectra(data2)
            if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
                var2 = h.get_converter_array(kwargs['z_converter'])[0](var2)
            second_data_set = True
            noise_levels = False

        elif type(args[0]) == np.ndarray:
            second_data_set = False
            noise_levels = True
    else:
        second_data_set = False
        noise_levels = False

    # plot spectra
    ifig = 1
    n_figs = n_time * n_height

    for iTime in range(n_time):
        for iHeight in range(n_height):
            fig, ax = plt.subplots(1, figsize=fig_size)

            dTime = h.ts_to_dt(time[iTime])
            rg = height[iHeight]

            ax.text(0.01, 0.93,
                    '{} UTC  at {:.2f} m ({})'.format(dTime.strftime("%Y-%m-%d %H:%M:%S"), rg,
                                                      data['system']),
                    horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.step(vel, var[iTime, iHeight, :], color='blue', linestyle='-',
                    linewidth=2, label=data['system'] + ' ' + data['name'])

            # if a 2nd dict is given, assume another dataset and plot on top
            if second_data_set:
                # find the closest spectra to the first device
                iTime2 = h.argnearest(time2, time[iTime])
                iHeight2 = h.argnearest(height2, rg)

                dTime2 = h.ts_to_dt(time2[iTime2])
                rg2 = height2[iHeight2]

                ax.text(0.01, 0.85,
                        '{} UTC  at {:.2f} m ({})'.format(dTime2.strftime("%Y-%m-%d %H:%M:%S"), rg, data2['system']),
                        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

                ax.step(vel2, var2[iTime2, iHeight2, :], color='orange', linestyle='-',
                        linewidth=2, label=data2['system'] + ' ' + data2['name'])

            if noise_levels:
                mean = h.lin2z(args[0][iTime, iHeight, 0])
                thresh = h.lin2z(args[0][iTime, iHeight, 1])

                # plot mean noise line and threshold
                x1, x2 = vel[0], vel[-1]
                ax.plot([x1, x2], [thresh, thresh], color='k', linestyle='-', linewidth=1, label='noise threshold')
                ax.plot([x1, x2], [mean, mean], color='k', linestyle='--', linewidth=1, label='mean noise')

                ax.text(0.01, 0.85,
                        'noise floar threshold = {:.2f} \nmean noise floar =  {:.2f} '.format(thresh, mean),
                        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

            ax.set_xlim(left=velmin, right=velmax)
            ax.set_ylim(bottom=vmin, top=vmax)
            ax.set_xlabel('Doppler Velocity (m/s)', fontweight='semibold', fontsize=fsz)
            ax.set_ylabel('Reflectivity (dBZ)', fontweight='semibold', fontsize=fsz)
            ax.grid(linestyle=':')

            ax.legend(fontsize=fsz)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            if 'save' in kwargs:
                figure_name = name + '{}_{}_{:5.2f}m.png'.format(str(ifig).zfill(4),
                                                                 dTime.strftime('%Y%m%d_%H%M%S_UTC'),
                                                                 height[iHeight])
                fig.savefig(figure_name, dpi=150)
                print("   Saved {} of {} png to  {}".format(ifig, n_figs, figure_name))

            ifig += 1
            if ifig != n_figs + 1: plt.close(fig)

    return fig, ax
