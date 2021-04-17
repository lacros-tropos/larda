#!/usr/bin/python3

import datetime
import numpy as np
import netCDF4
import ast
import sys
sys.path.append('../')
sys.path.append('.')
import pyLARDA.helpers as h
import pyLARDA.Transformations as Transf
import pyLARDA.NcReader as NcReader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def trace_reader(paraminfo):
    """build a function for reading the trace_airmass_source data (setup by connector)"""
    def t_r(f, time_interval, *further_intervals):
        """function that converts the trace_airmass_source netCDF to the data container
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            #print('timestamps ', ts[:5])
            # setup slice to load base on time_interval
            it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
            if len(time_interval) == 2:
                it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))
                if ts[it_e] < h.dt_to_ts(time_interval[0])-3*np.median(np.diff(ts)):
                    logger.warning(
                            'last profile of file {}\n at {} too far from {}'.format(
                                f, h.ts_to_dt(ts[it_e]), time_interval[0]))
                    return None

                it_e = it_e+1 if not it_e == ts.shape[0]-1 else None
                slicer = [slice(it_b, it_e)]
            else:
                slicer = [slice(it_b, it_b+1)]
            print(slicer)

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
                    ir_e = ir_e+1 if not ir_e == ranges.shape[0]-1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b+1))

            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'])

            its = np.arange(ts.shape[0])[tuple(slicer)[0]]
            irs = np.arange(ranges.shape[0])[tuple(slicer)[1]]
            var = np.empty((its.shape[0], irs.shape[0]))
            mask = np.empty((its.shape[0], irs.shape[0]))
            mask[:] = False

            no_occ = ncD.variables[paraminfo['variable_name'] + "_no_below"][tuple(slicer)[0],tuple(slicer)[1]]
            occ_height = ncD.variables[paraminfo['variable_name']][tuple(slicer)[0],tuple(slicer)[1],:]
            var = occ_height*no_occ[:, :, np.newaxis]

            data = {}
            data['dimlabel'] = ['time', 'range', 'cat']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = NcReader.get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            variable = ncD.variables[paraminfo['variable_name']]
            var_definition = ast.literal_eval(
                variable.getncattr(paraminfo['identifier_var_def']))
            if var_definition[1] == "forrest":
                var_definition[1] = "forest"

            data['var_definition'] = var_definition

            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = NcReader.get_var_attr_from_nc("identifier_rg_unit", 
                                                paraminfo, ranges)
            logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))

            data['var_unit'] = NcReader.get_var_attr_from_nc("identifier_var_unit", 
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                NcReader.get_var_attr_from_nc("identifier_var_lims", 
                                                    paraminfo, var)]

            data['var'] = varconverter(var)
            data['mask'] = maskconverter(mask)

            return data

    return t_r


def plot_ls_2d(airmass_source, plottop=12000, xright=7000, xlbl_time='%H:%M', norm=None):
    #f, parameter, dt_list, dsp, config, savepath, config_dict, model):
    time_list = airmass_source['ts']
    dt_list = [h.ts_to_dt(time) for time in time_list]
    height_list = airmass_source['rg']/1000.
    no_plots = len(dt_list)

    if no_plots > 9:
        xsize = 15
    elif no_plots < 8:
        xsize = no_plots*1.8
    else:
        xsize = 12
    
    if norm:
        airmass_source['var'][:,:,:] = airmass_source['var'][:,:,:].copy()/norm
        dsp_text = 'Norm. residence time'
    else:
        dsp_text = 'acc. residence time'

    axes = []
    fig = plt.figure(constrained_layout=False,
                     figsize=(xsize, 6))
    gs1 = fig.add_gridspec(nrows=9, ncols=no_plots, wspace=0.0)
    for i in range(no_plots):
        axes.append(fig.add_subplot(gs1[1:, i]))
    #fig, axes = plt.subplots(1, no_plots, sharex=True, sharey=True, figsize=(9, 6))
    ls_colors = ['lightskyblue', 'darkgreen', 'khaki', 'palegreen', 'red', 'gray', 'tan']
    ls_colors = ['lightskyblue', 'seagreen', 'khaki', '#6edd6e', 'darkmagenta', 'gray', 'tan']

    for it, dt in enumerate(dt_list):

        occ_height = airmass_source['var'][it,:,:]
        occ_left = np.cumsum(occ_height, axis=1)

        categories = airmass_source['var_definition']

        l1 = axes[it].barh(height_list, occ_height[:, 0].T, 
                           align='center', height=0.35, color=ls_colors[0], edgecolor='none')
        l2 = axes[it].barh(height_list, occ_height[:, 1].T, left=occ_left[:, 0].T,
                           align='center', height=0.35, color=ls_colors[1], edgecolor='none')
        l3 = axes[it].barh(height_list, occ_height[:, 2].T, left=occ_left[:, 1].T,
                           align='center', height=0.35, color=ls_colors[2], edgecolor='none')
        l4 = axes[it].barh(height_list, occ_height[:, 3].T, left=occ_left[:, 2].T,
                           align='center', height=0.35, color=ls_colors[3], edgecolor='none')
        l5 = axes[it].barh(height_list, occ_height[:, 4].T, left=occ_left[:, 3].T,
                           align='center', height=0.35, color=ls_colors[4], edgecolor='none')
        l6 = axes[it].barh(height_list, occ_height[:, 5].T, left=occ_left[:, 4].T,
                           align='center', height=0.35, color=ls_colors[5], edgecolor='none')

        l7 = axes[it].barh(height_list, occ_height[:, 6].T, left=occ_left[:, 5].T,
                          align='center', height=0.35, color=ls_colors[6], edgecolor='none')

        #axes[it].set_ylim([0, 12])
        axes[it].set_ylim([0, plottop/1000.])
        axes[it].tick_params(axis='y', which='major', labelsize=14, 
                             width=1.5, length=3)
        axes[it].tick_params(axis='both', which='minor', width=1, length=2)
        axes[it].tick_params(axis='both', which='both', right=True, top=True,
                             direction='in')
        axes[it].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1.0))
        axes[it].set_xlabel(dt.strftime(xlbl_time), fontsize=14)
        
        axes[it].set_xlim(right=xright)
        if xright < 20000:
            axes[it].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3000))
        else:
            axes[it].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=1))
        if norm:
            axes[it].xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        axes[it].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axes[it].tick_params(axis='x', labeltop=False, labelbottom=False)
        axes[it].tick_params(axis='y', labelleft=False)

    axes[0].set_ylabel("Height [km]", fontweight='semibold', fontsize=15)
    axes[0].tick_params(axis='y', labelleft=True)
    axes[-1].tick_params(axis='x', labeltop=True, labelbottom=False, labelsize=13)
    
    plt.suptitle("{}   {}   {}".format(dt.strftime("%Y%m%d"), airmass_source['paraminfo']['location'], airmass_source['name']), 
                 fontweight='semibold', fontsize=15)


    axes[-1].annotate("Endpoint: {:.1f} {:.1f} ".format(airmass_source['paraminfo']['coordinates'][1], 
                                                        airmass_source['paraminfo']['coordinates'][0]), 
                        xy=(.88, 0.92), xycoords='figure fraction',
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=12)
    axes[0].annotate('Time UTC', xy=(.5, .02),
                     xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='bottom',
                     fontsize=15, fontweight='semibold')

    axes[-1].annotate(dsp_text, xy=(.87, 0.845),
                      xycoords='figure fraction',
                      horizontalalignment='center', verticalalignment='bottom',
                      fontsize=13, fontweight='semibold')
    fig.legend((l1, l2, l3, l4, l5, l6, l7), list(categories.values()),
               #bbox_to_anchor=(0.85, 0.952),
               bbox_to_anchor=(0.07, 0.94), loc='upper left',
               columnspacing=1,
               ncol=4, fontsize=14, framealpha=0.8)

    #fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.1, 'h_pad': 1.5})
    #plt.tight_layout(w_pad=0.0002)

    
    return fig, axes


def plot_gn_2d(airmass_source, plottop=12000, xright=7000, xlbl_time='%H:%M', norm=None):
    #f, parameter, dt_list, dsp, config, savepath, config_dict, model):
    time_list = airmass_source['ts']
    dt_list = [h.ts_to_dt(time) for time in time_list]
    height_list = airmass_source['rg']/1000.
    no_plots = len(dt_list)

    if no_plots > 9:
        xsize = 15
    elif no_plots < 8:
        xsize = no_plots*1.8
    else:
        xsize = 12


    if norm:
        airmass_source['var'][:,:,:] = airmass_source['var'][:,:,:].copy()/norm
        dsp_text = 'Norm. residence time'
    else:
        dsp_text = 'acc. residence time'
        
#     fig, axes = plt.subplots(1, no_plots, sharex=True, sharey=True, 
#                              figsize=(xsize, 6))
    axes = []
    fig = plt.figure(constrained_layout=False,
                     figsize=(xsize, 6))
    gs1 = fig.add_gridspec(nrows=9, ncols=no_plots, wspace=0.0)
    for i in range(no_plots):
        axes.append(fig.add_subplot(gs1[1:, i]))
    

    colors = [(0.65098039215686276, 0.84705882352941175, 0.32941176470588235, 1.0), 
              (1.0, 0.85098039215686272, 0.18431372549019609, 1.0), 
              (0.89803921568627454, 0.7686274509803922, 0.58039215686274515, 1.0),
              (0.40000000000000002, 0.76078431372549016, 0.6470588235294118, 1.0), 
              (0.9882352941176471, 0.55294117647058827, 0.3843137254901961, 1.0), 
              (0.55294117647058827, 0.62745098039215685, 0.79607843137254897, 1.0),  
              (0.70196078431372544, 0.70196078431372544, 0.70196078431372544, 1.0)]

    for it, dt in enumerate(dt_list):

        occ_height = airmass_source['var'][it,:,:]
        occ_left = np.cumsum(occ_height, axis=1)

        geo_names = airmass_source['var_definition']

        l = []
        for i in range(len(geo_names)):
            if i == 0:
                l.append(axes[it].barh(height_list, occ_height[:, 0].T, left=0, 
                    align='center', height=0.3, color=colors[0], edgecolor='none'))
            else:
                l.append(axes[it].barh(height_list, occ_height[:, i].T, left=occ_left[:, i-1].T,
                    align='center', height=0.3, color=colors[i], edgecolor='none'))

        #axes[it].set_ylim([0, 12])
        axes[it].set_ylim([0, plottop/1000.])
        axes[it].tick_params(axis='y', which='major', labelsize=14, 
                             width=1.5, length=3)
        axes[it].tick_params(axis='both', which='minor', width=1, length=2)
        axes[it].tick_params(axis='both', which='both', right=True, top=True,
                             direction='in')
        axes[it].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1.0))
        axes[it].set_xlabel(dt.strftime(xlbl_time), fontsize=14)
        
        axes[it].set_xlim([0, xright])        
        if xright < 20000:
            axes[it].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3000))
        else:
            # axes[it].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, xright/2.]))
            axes[it].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=1))
        if norm:
            axes[it].xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        axes[it].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axes[it].tick_params(axis='x', labeltop=False, labelbottom=False)
        axes[it].tick_params(axis='y', labelleft=False)

    axes[0].set_ylabel("Height [km]", fontweight='semibold', fontsize=15)
    axes[0].tick_params(axis='y', labelleft=True)
    axes[-1].tick_params(axis='x', labeltop=True, labelbottom=False, labelsize=13)
    
    plt.suptitle("{}   {}   {}".format(dt.strftime("%Y%m%d"), airmass_source['paraminfo']['location'], airmass_source['name']), 
                 fontweight='semibold', fontsize=15)


    axes[-1].annotate("Endpoint: {:.1f} {:.1f} ".format(airmass_source['paraminfo']['coordinates'][1], 
                                                        airmass_source['paraminfo']['coordinates'][0]), 
                        xy=(.88, 0.92), xycoords='figure fraction',
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=12)
    axes[0].annotate('Time UTC', xy=(.5, .02),
                     xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='bottom',
                     fontsize=15, fontweight='semibold')

    axes[-1].annotate(dsp_text, xy=(.87, 0.845),
                      xycoords='figure fraction',
                      horizontalalignment='center', verticalalignment='bottom',
                      fontsize=13, fontweight='semibold')
    fig.legend(l, list(geo_names.values()),
               loc='upper left',
               #bbox_to_anchor=(0.01, 0.952),
               bbox_to_anchor=(0.07, 0.94),
               columnspacing=1,
               ncol=4, fontsize=14, framealpha=0.8)

    return fig, axes



def trace_reader2(paraminfo):
    """build a function for reading the new trace format (since Dec 2020) data (setup by connector)"""
    def t_r(f, time_interval, *further_intervals):
        """function that converts the trace netCDF to the data container
        """
        logger.debug("filename at reader {}".format(f))
        with netCDF4.Dataset(f, 'r') as ncD:

            times = ncD.variables[paraminfo['time_variable']][:].astype(np.float64)

            timeconverter, _ = h.get_converter_array(
                paraminfo['time_conversion'], ncD=ncD)
            ts = timeconverter(times)

            #print('timestamps ', ts[:5])
            # setup slice to load base on time_interval
            it_b = h.argnearest(ts, h.dt_to_ts(time_interval[0]))
            if len(time_interval) == 2:
                it_e = h.argnearest(ts, h.dt_to_ts(time_interval[1]))
                if ts[it_e] < h.dt_to_ts(time_interval[0])-3*np.median(np.diff(ts)):
                    logger.warning(
                            'last profile of file {}\n at {} too far from {}'.format(
                                f, h.ts_to_dt(ts[it_e]), time_interval[0]))
                    return None

                it_e = it_e+1 if not it_e == ts.shape[0]-1 else None
                slicer = [slice(it_b, it_e)]
            else:
                slicer = [slice(it_b, it_b+1)]
            print(slicer)

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
                    ir_e = ir_e+1 if not ir_e == ranges.shape[0]-1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b+1))

            varconverter, maskconverter = h.get_converter_array(
                paraminfo['var_conversion'])

            its = np.arange(ts.shape[0])[tuple(slicer)[0]]
            irs = np.arange(ranges.shape[0])[tuple(slicer)[1]]
            var = np.empty((its.shape[0], irs.shape[0]))
            mask = np.empty((its.shape[0], irs.shape[0]))
            mask[:] = False

            var = ncD.variables[paraminfo['variable_name']][tuple(slicer)[0],tuple(slicer)[1],:]

            data = {}
            data['dimlabel'] = ['time', 'range', 'cat']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = NcReader.get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            variable = ncD.variables[paraminfo['variable_name']]
            var_definition = ast.literal_eval(
                variable.getncattr(paraminfo['identifier_var_def']))
            if var_definition[1] == "forrest":
                var_definition[1] = "forest"

            data['var_definition'] = var_definition

            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = NcReader.get_var_attr_from_nc("identifier_rg_unit", 
                                                paraminfo, ranges)
            logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))

            data['var_unit'] = NcReader.get_var_attr_from_nc("identifier_var_unit", 
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                NcReader.get_var_attr_from_nc("identifier_var_lims", 
                                                    paraminfo, var)]

            data['var'] = varconverter(var)
            data['mask'] = maskconverter(mask)

            return data

    return t_r
