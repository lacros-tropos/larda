#!/usr/bin/python3


import datetime
#import itertools
import copy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate

import pyLARDA.VIS_Colormaps as VIS_Colormaps
import pyLARDA.helpers as h


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
    assert datadict1['type'] == datadict2['type']
    new_data['type'] = datadict1['type']
    container_type = datadict1['type']
    
    if container_type == "timeheight":
        print(datadict1['ts'].shape, datadict1['rg'].shape, datadict1['var'].shape)
        print(datadict2['ts'].shape, datadict2['rg'].shape, datadict2['var'].shape)
    thisjoint = datadict1['ts'].shape[0]
    new_data["joints"] = datadict1.get('joints',[]) + [thisjoint] + datadict2.get('joints',[])
    print("joints", new_data['joints'])
    new_data['filename'] = h.flatten([datadict1['filename']] + [datadict2['filename']])
    print(new_data['filename'])

    assert datadict1['paraminfo'] == datadict2['paraminfo']
    new_data['paraminfo'] = datadict1['paraminfo']
    if container_type == "timeheight":
        assert datadict1['rg_unit'] == datadict2['rg_unit']
        new_data['rg_unit'] = datadict1['rg_unit']
        assert datadict1['colormap'] == datadict2['colormap']
        new_data['colormap'] = datadict1['colormap']
        assert np.all(datadict1['rg'] == datadict2['rg']), (datadict1['rg'], datadict2['rg'])
    assert datadict1['var_unit'] == datadict2['var_unit']
    new_data['var_unit'] = datadict1['var_unit']
    assert datadict1['var_lims'] == datadict2['var_lims']
    new_data['var_lims'] = datadict1['var_lims']
    assert datadict1['system'] == datadict2['system']
    new_data['system'] = datadict1['system']
    assert datadict1['name'] == datadict2['name']
    new_data['name'] = datadict1['name']
    print(new_data['type'])
    print(new_data['paraminfo'])


    if container_type == "timeheight" \
            or container_type == "timeheightspec":
        new_data['rg'] = datadict1['rg']
        new_data['ts'] = np.hstack((datadict1['ts'], datadict2['ts']))
        new_data['var'] = np.vstack((datadict1['var'], datadict2['var']))
        new_data['mask'] = np.vstack((datadict1['mask'], datadict2['mask']))
        print(new_data['ts'].shape, new_data['rg'].shape, new_data['var'].shape)
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
    print('var min', data['var'][~data['mask']].min())
    print(var)

    kx, ky = 1, 1
    interp_var = scipy.interpolate.RectBivariateSpline(
        data['ts'], data['rg'], var, 
        kx=kx, ky=ky)
    interp_mask = scipy.interpolate.RectBivariateSpline(
        data['ts'], data['rg'], data['mask'].astype(np.float),
        kx=kx, ky=ky)
    print(data['mask'].astype(np.float)[:,10])
    print(data['mask'][:,10])

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']
    new_var = interp_var(new_time, new_range, grid=True)
    new_mask = interp_mask(new_time, new_range, grid=True)

    print('new_mask', new_mask)
    new_mask[new_mask > mask_thres] = 1
    new_mask[new_mask < mask_thres] = 0
    print('new_mask', new_mask)

    print(new_var.shape, new_var)
    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = new_var
    interp_data['mask'] = new_mask
    print(new_time.shape, new_range.shape, new_var.shape, new_mask.shape)

    return interp_data


def combine(func, datalist, keys_to_update, **kwargs):
    """apply a func to the variable

    Args:
        func: a function that takes [datacontainer1, datacontainer2, ..]
            as given input (order as given in datalist)
        datalist: list of data containers
        keys_to_update: dictionary of keys to update
    """
    
    if len(datalist) > 1:
        assert np.all(datalist[0]['rg'] == datalist[1]['rg']) 
        assert np.all(datalist[0]['ts'] == datalist[1]['ts'])

    #use the first dict as the base
    new_data = {**datalist[0]}
    new_data.update(keys_to_update)

    new_data['var'], new_data['mask'] = func(datalist)
    new_data['history'] = {
        'filename': [e['filename'] for e in datalist],
        'paraminfo': [e['paraminfo'] for e in datalist],
    }
    
    return new_data


def plot(data):
    """call correct function based on type"""


def plottimeseries(data, **kwargs):
    """plot a timeheight data container

    Args:
        data (dict): data container
        **time_interval (list dt): constrain plot to this dt
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
    """
    assert data['type'] == 'timeseries', 'wrong plot function for {}'.format(data['type'])
    time_list = data['ts']
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    # this is the last valid index
    var = var.filled(-999)
    jumps = np.where(np.diff(time_list)>60)[0]
    for ind in jumps[::-1].tolist():
        print("jump at ", ind, dt_list[ind-1:ind+2])
        # and modify the dt_list
        dt_list.insert(ind+1, dt_list[ind]+datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind+1, -999, axis=0)
    
    var = np.ma.masked_equal(var, -999) 

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    vmin, vmax = data['var_lims']
    print("varlims", vmin,vmax)
    if 'z_converter' in kwargs: 
        if kwargs['z_converter'] == 'log':
            #plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            ax.set_yscale('log')
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    
    ax.plot(dt_list, var)

    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    ax.set_ylim([vmin, vmax])
    
    #ax.set_ylim([height_list[0], height_list[-1]])
    #ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    #ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)

    ylabel = "{} {} [{}]".format(data["system"], data["name"], data['var_unit']) 
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    time_extend = dt_list[-1] - dt_list[0]
    print(time_extend)
    if time_extend > datetime.timedelta(hours=6):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    elif time_extend > datetime.timedelta(hours=1):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,15,30,45]))
    else:
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,15,30,45]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,5,10,15,20,25,30,35,40,45,50,55]))

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14, 
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)

    return fig, ax



def plot2d(data, **kwargs):
    """plot a timeheight data container

    Args:
        data (dict): data container
        **time_interval (list dt): constrain plot to this dt
        **range_interval (list float): constrain plot to this ranges
        **z_converter (string): convert var before plotting
                use eg 'lin2z' or 'log'
        **contour: add a countour
    """
    assert data['type'] == 'timeheight', 'wrong plot function for {}'.format(data['type'])
    time_list = data['ts']
    range_list = data['rg']
    var = np.ma.masked_where(data['mask'], data['var']).copy()
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
    # this is the last valid index
    var = var.filled(-999)
    jumps = np.where(np.diff(time_list)>60)[0]
    for ind in jumps[::-1].tolist():
        print("jump at ", ind, dt_list[ind-1:ind+2])
        # and modify the dt_list
        dt_list.insert(ind+1, dt_list[ind]+datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind+1, np.full(range_list.shape, -999), axis=0)
    
    var = np.ma.masked_equal(var, -999) 

    vmin, vmax = data['var_lims']
    print("varlims", vmin,vmax)
    plotkwargs = {}
    if 'z_converter' in kwargs: 
        if kwargs['z_converter'] == 'log':
            plotkwargs['norm'] = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            var = h.get_converter_array(kwargs['z_converter'])[0](var)
    colormap = data['colormap']
    print("custom colormaps ", VIS_Colormaps.custom_colormaps.keys())
    if colormap in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap]
    
    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]),
                           range_list[:],
                           np.transpose(var[:, :]),
                           #cmap=VIS_Colormaps.carbonne_map, 
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
        ax.clabel(cont, fontsize=10, inline=1, fmt='%1.1f',)

    cbar = fig.colorbar(pcmesh, fraction=0.13)
    if 'time_interval' in kwargs.keys():
        ax.set_xlim(kwargs['time_interval'])
    if 'range_interval' in kwargs.keys():
        ax.set_ylim(kwargs['range_interval'])
    
    #ax.set_ylim([height_list[0], height_list[-1]])
    #ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    #ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ylabel = 'Height [{}]'.format(data['rg_unit'])
    ax.set_ylabel(ylabel, fontweight='semibold', fontsize=15)

    z_string = "{} {} [{}]".format(data["system"], data["name"], data['var_unit']) 
    cbar.ax.set_ylabel(z_string, fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    time_extend = dt_list[-1] - dt_list[0]
    print(time_extend)
    if time_extend > datetime.timedelta(hours=6):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    elif time_extend > datetime.timedelta(hours=1):
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,15,30,45]))
    else:
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,15,30,45]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,5,10,15,20,25,30,35,40,45,50,55]))

    #ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())


    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14, 
                   width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)
    cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)

    plt.subplots_adjust(right=0.99)
    return fig, ax



def scatter(axh, text, x, y, marker, x_min, x_max, y_min, y_max, x_lab, y_lab):
    import matplotlib.ticker as ticker
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # place_text(axh, [.05, 1.1], text )
    axh.plot(x, y, marker)
    axh.set_xlabel(x_lab)
    axh.set_ylabel(y_lab)
    axh.set_xlim(left=x_min, right=x_max)
    axh.set_ylim(bottom=y_min, top=y_max)
    axh.xaxis.set_ticks((np.arange(x_min, x_max, (x_max - x_min) / 4.0)))
    axh.yaxis.set_ticks((np.arange(y_min, y_max, (y_max - y_min) / 4.0)))
    axh.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    axh.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    axh.grid(linestyle=':')
    # axh.legend(loc="upper right")
    axh.set_aspect('equal', 'box')

    # plot 1:1 line
    N = 100
    X = np.linspace(x_min, x_max, N)
    axh.plot(X, X, 'k--')

    # add patches
    color1 = [np.divide([31, 119, 180], 255.), np.divide([255, 127, 14], 255.)]
    patches = [Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_max]], facecolor='C0', fill=True),
               Polygon([[x_min, y_min], [x_min, y_max], [x_max, y_max]], facecolor='C1', fill=True)]

    p = PatchCollection(patches, alpha=0.25)
    p.set_color(color1)
    axh.add_collection(p)

