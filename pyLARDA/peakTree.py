#!/usr/bin/python3

import datetime
import numpy as np
import netCDF4
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


try:
    # compile and load the c-version
    #import pyximport
    #pyximport.install()
    import pyLARDA.peakTree_fastbuilder as peakTree_fastbuilder
    fastbuilder = True
except:
    # use the numpy only version
    fastbuilder = False
    #logger.exception("compiling peakTree_fastbuilder failed, using numpy version")

def build_tree_py(data, ldr_avail):
    """pure python/numpy version of the build tree function
    (slower than the compiled cython version)

    Indices in the stacked data array:

    .. code::
    
        0: 'parent', 1: 'Z', 2: 'v',
        3: 'width', 4: 'skew', 5: 'threshold',
        6: 'prominence', 7: 'bound_l', 8: 'bound_r',
        optional
        9: 'LDR', 10: 'ldrmax'
    """
    
    travtree = {}            
    #logger.debug('peakTree parent {}'.format(ncD.variables['parent'][it,ir,:]))
    parent = np.ma.masked_less(data[:,0], -990)
    avail_nodes = np.argwhere(parent > -10).ravel()
    #print(data[:,0].mask, type(data[:,0]), parent, avail_nodes)
    for k in avail_nodes.tolist():
        node = {'parent_id': np.asscalar(data[k,0]), 
                'thres': np.asscalar(data[k,5]), 
                'width': np.asscalar(data[k,3]), 
                'z': np.asscalar(data[k,1]), 
                'bounds': (np.asscalar(data[k,7]), np.asscalar(data[k,8])),
                #'coords': [0], 
                'skew': np.asscalar(data[k,4]),
                'prominence': np.asscalar(data[k,6]),
                'v': np.asscalar(data[k,2])}
        node['id'] = k
        node['bounds'] = list(map(int, node['bounds']))
        node['width'] = node['width'] if np.isfinite(node['width']) else -99
        node['skew'] = node['skew'] if np.isfinite(node['skew']) else -99
        node['thres'] = node['thres'] if np.isfinite(node['thres']) else -99
        node['prominence'] = node['prominence'] if np.isfinite(node['prominence']) else -99
        if ldr_avail:
            node['ldr'] = np.asscalar(data[k,9]) 
            node['ldr'] = node['ldr'] if np.isfinite(node['ldr']) else -99
            node['ldrmax'] = np.asscalar(data[k,10])
            node['ldrmax'] = node['ldrmax'] if np.isfinite(node['ldrmax']) else -99
        else:
            node['ldr'], node['ldrmax'] = -99, -99
        if node['parent_id'] != -999:
            if k == 0:
                node['coords'] = [0]
            else:
                coords = travtree[node['parent_id']]['coords']
                if k%2 == 0:
                    node['coords'] = coords + [1]
                else:
                    node['coords'] = coords + [0]
            
            # remove the parent id for compatibility to peakTreeVis
            if node['parent_id'] == -1:
                del node['parent_id']
            # format for transport
            #v = {ky: format_for_json(val) for ky, val in v.items()}
            travtree[k] = node
    return travtree


def array_to_tree_py(data, ldr_avail):
    """convert the array from the netcdf to var of data container
    pure python/numpy version
    (slower than the compiled cython version)
    """
    trees = np.empty((data.shape[0], data.shape[1]), dtype=object)
    mask = np.empty((data.shape[0], data.shape[1]), dtype=bool)
    mask[:] = True
    
    for it in range(data.shape[0]):
        for ir in range(data.shape[1]):
            trees[it, ir] = build_tree_py(data[it,ir,:,:], ldr_avail)
            mask[it, ir] = False
            
    return trees, mask


def peakTree_reader(paraminfo):
    """build a function for reading the peakTree data (setup by connector)"""
    def pt_ret(f, time_interval, *further_intervals):
        """function that converts the peakTree netCDF to the data container
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

            if paraminfo['ncreader'] == 'peakTree':
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
            var = np.empty((its.shape[0], irs.shape[0]), dtype=object)
            mask = np.empty((its.shape[0], irs.shape[0]), dtype=bool)
            mask[:] = True

            param_list = [
                ncD.variables['parent'][tuple(slicer)[0],tuple(slicer)[1],:], #0
                ncD.variables['Z'][tuple(slicer)[0],tuple(slicer)[1],:],      #1
                ncD.variables['v'][tuple(slicer)[0],tuple(slicer)[1],:],      #2
                ncD.variables['width'][tuple(slicer)[0],tuple(slicer)[1],:],  #3
                ncD.variables['skew'][tuple(slicer)[0],tuple(slicer)[1],:],   #4
                ncD.variables['threshold'][tuple(slicer)[0],tuple(slicer)[1],:], #5
                ncD.variables['prominence'][tuple(slicer)[0],tuple(slicer)[1],:], #6
                ncD.variables['bound_l'][tuple(slicer)[0],tuple(slicer)[1],:],    #7
                ncD.variables['bound_r'][tuple(slicer)[0],tuple(slicer)[1],:]     #8
            ]
            if 'LDR' in ncD.variables.keys():
                ldr_avail = True
                param_list.append(ncD.variables['LDR'][tuple(slicer)[0],tuple(slicer)[1],:])  #9
                param_list.append(ncD.variables['ldrmax'][tuple(slicer)[0],tuple(slicer)[1],:]) #10
            else:
                ldr_avail = False
            data = np.stack(tuple(param_list), axis=3)
            print(data.shape)
            if fastbuilder:
                var, mask = peakTree_fastbuilder.array_to_tree_c(data.astype(float), ldr_avail)
            else:
                var, mask = array_to_tree_py(data, ldr_avail)

            data = {}
            data['dimlabel'] = ['time', 'range', 'dict']

            data["filename"] = f
            data["paraminfo"] = paraminfo
            data['ts'] = ts[tuple(slicer)[0]]

            data['system'] = paraminfo['system']
            data['name'] = paraminfo['paramkey']
            data['colormap'] = paraminfo['colormap']

            if 'meta' in paraminfo:
                data['meta'] = NcReader.get_meta_from_nc(ncD, paraminfo['meta'], paraminfo['variable_name'])

            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = NcReader.get_var_attr_from_nc("identifier_rg_unit", 
                                                paraminfo, ranges)
            logger.debug('shapes {} {} {}'.format(ts.shape, ranges.shape, var.shape))

            logger.debug('shapes {} {}'.format(ts.shape, var.shape))
            data['var_unit'] = NcReader.get_var_attr_from_nc("identifier_var_unit", 
                                                    paraminfo, var)
            data['var_lims'] = [float(e) for e in \
                                NcReader.get_var_attr_from_nc("identifier_var_lims", 
                                                    paraminfo, var)]

            data['var'] = varconverter(var)
            data['mask'] = maskconverter(mask)

            return data

    return pt_ret


def tree_to_timeheight(data_cont, param, sel_node=0, **kwargs):
    """convert the tree data container to a normal time-height data container by extracting a node and a parameter
    
    Args:
        data_cont (dict): data container
        param (str): parameter to select, eg z, v
        sel_node (np.array or int, optional): integer or array of index to select
        **kwargs


    Returns:
        data container of dimension ``['time', 'range']``
    """
    assert data_cont['dimlabel'] == ['time', 'range', 'dict'], "dimlabel does not match"
    if type(sel_node) is not int:
        assert data_cont['var'].shape == sel_node.shape
        sel_nodes_array = True
    else:
        sel_nodes_array = False
    
    new_cont = data_cont.copy()
    var = np.empty(data_cont['var'].shape, dtype=float)
    var[:] = np.nan
    mask = np.empty(data_cont['var'].shape, dtype=bool)
    mask[:] = True
    for index, tree in np.ndenumerate(data_cont['var']):
        #print(index, sel_nodes[index])
        if param == 'no_nodes':
            var[index] = len(data_cont['var'][index].keys())
            mask[index] = False
        elif sel_nodes_array:
            if not sel_node[index] == -1:
                var[index] = data_cont['var'][index][sel_node[index]][param]
                mask[index] = False
        else:
            if sel_node in data_cont['var'][index]:
                var[index] = data_cont['var'][index][sel_node][param]
                mask[index] = False
            
            
    new_cont['var'] = var
    new_cont['mask'] = mask
    new_cont['dimlabel'] = ['time', 'range']
    new_cont['colormap'] = 'jet'
    new_cont['var_lims'] = [-50, 10]
    if 'var_units' in kwargs:
        new_cont['var_units'] = kwargs
    return new_cont


def select_rimed_node(data_cont):
    """select the rimed nodes from a peaktree data container
    The nodes are filtered by the rule:
    ``abs(n['v'][0]-n['v'][-1]) > 1.5``

    Args:
        data_cont: peakTree data container

    Returns:
        data_container with selected indices in ``var`` of shape ``(time, range)``
    """

    new_cont = {**data_cont}
    var = np.empty(data_cont['var'].shape, dtype=int)
    var[:] = -1
    for index, tree in np.ndenumerate(data_cont['var']):
        nodes = list(tree.values())
        if nodes:
            nodes.sort(key=lambda n: n['v'])
            if len(nodes) > 1:
                if abs(nodes[0]['v'] - nodes[-1]['v']) > 1.5:
                    var[index] = nodes[0]['id']
                #print(index, nodes)
            #print(index, len(nodes))
            #print(k, ' filtered nodes ', len(nodes), [(e['id'], e['z'], e['v']) for e in nodes])

    new_cont['var'] = var
    new_cont['mask'] = (var == -1)
    new_cont['name'] = 'selected index'
    new_cont['dimlabel'] = ['time', 'range']
    new_cont['var_unit'] = ''
    return new_cont


def select_liquid_node(data_cont, **kwargs):
    """select the liquid nodes from a peaktree data container

    The nodes are filtered by the rule:
    ``n['z'] < -20 and abs(n['v']) < 0.3``

    Args:
        data_cont: peakTree data container

    Key word arguments:
        Z_thresh, maximum Z of liquid peak (default is -20)
        LDR_thresh, maximum LDR of liquid peaks (LDR ignored if not given)

    Returns:
        data_container with selected indices in ``var`` of shape ``(time, range)``
    """
    Z_thresh = kwargs['Z_thresh'] if 'Z_thresh' in kwargs else -20
    LDR_thresh = kwargs['LDR_thresh'] if 'LDR_thresh' in kwargs else False
    new_cont = {**data_cont}
    var = np.empty(data_cont['var'].shape, dtype=int)
    var[:] = -1
    for index, tree in np.ndenumerate(data_cont['var']):
        if not LDR_thresh:
            nodes = list(filter(lambda n: n['z'] < Z_thresh and abs(n['v']) < 0.3, tree.values()))
        else:
            nodes = list(filter(lambda n: n['z'] < Z_thresh and abs(n['v']) < 0.3 and n['ldr'] < LDR_thresh,
                                tree.values()))
        if nodes:
            nodes.sort(key=lambda n: n['v'])
            if len(nodes) > 1:
                pass
                # print(index, nodes)
            # print(index, len(nodes))
            # print(k, ' filtered nodes ', len(nodes), [(e['id'], e['z'], e['v']) for e in nodes])
            var[index] = nodes[-1]['id']

    new_cont['var'] = var
    new_cont['mask'] = (var == -1)
    new_cont['name'] = 'selected index'
    new_cont['dimlabel'] = ['time', 'range']
    new_cont['var_unit'] = ''
    return new_cont


def select_fastest_node(data_cont):
    """select the fastest-falling nodes from a peaktree data container

    Args:
        data_cont: peakTree data container

    Returns:
        data_container with selected indices in ``var`` of shape ``(time, range)``
    """

    new_cont = {**data_cont}
    var = np.empty(data_cont['var'].shape, dtype=int)
    var[:] = -1
    for index, tree in np.ndenumerate(data_cont['var']):
        if tree:
            fastest = min([x['v'] for x in tree.values()])
            nodes = list(filter(lambda n: n['v'] == fastest, tree.values()))
        else: nodes = []
        if nodes:
            nodes.sort(key=lambda n: n['id'])
            if len(nodes) > 1:
                pass
                # print(index, nodes)
            # print(index, len(nodes))
            # print(k, ' filtered nodes ', len(nodes), [(e['id'], e['z'], e['v']) for e in nodes])
            var[index] = nodes[0]['id']

    new_cont['var'] = var
    new_cont['mask'] = (var == -1)
    new_cont['name'] = 'selected index'
    new_cont['dimlabel'] = ['time', 'range']
    new_cont['var_unit'] = ''
    return new_cont


def plot_no_nodes(data_cont, **kwargs):
    """wrapper for :py:mod:`pyLARDA.Transformations.plot_timeheight2` to plot the no of nodes

    Args:
        data (dict): data container
        **kwargs: piped to plot_timeheight2 function

    Returns:
        ``fig, ax``
    """
    #data_cont['colormap'] = matplotlib.colors.ListedColormap(
    #    ["#ffffff", "#cdbfbc", "#987b61", "#fdff99", "#35d771", "#1177dd"], 'terrain_seq')
    data_cont['colormap'] = matplotlib.colors.ListedColormap(
    ["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat')
    # We must be sure to specify the ticks matching our target names
    #labels = {0: '0', 1: "1", 2: "3", 3: "5", 4: "7", 5: "9"}
    labels = {0: ' 0', 1: " 1", 2: " 2", 3: " 3", 4: " 4", 5: " 5"}

    data_cont['name'] = 'no. peaks'
    cbarformatter = plt.FuncFormatter(lambda val, loc: labels[val])
    data_cont['var'] = np.ceil(np.array(data_cont['var'])/2.)
    data_cont["var_lims"] = [-0.5, 5.5]
    fig, ax = Transf.plot_timeheight2(data_cont, **kwargs)
    #ax.cbar.set_yticks([0, 1, 2, 3, 4, 5])
    cbar = fig.axes[1]
    #cbar.set_ylabel("Number of nodes", fontweight='semibold', fontsize=15)
    #print(fig.axes[1].get_yticks())
    #print(cbar.ax.get_ticklocs())
    #print(fig.axes)
    #print(fig.axes[1])
    cbar_ylabel = ax.images[0].colorbar.ax.get_ylabel()
    ax.images[0].colorbar.ax.set_ylabel(cbar_ylabel[:-2])
    fig.axes[1].set_yticklabels(labels.values())
    return fig, ax


def plot_sel_index(data_cont, **kwargs):
    """wrapper for :py:mod:`pyLARDA.Transformations.plot_timeheight2` to plot the index of selected node

    Args:
        data (dict): data container
        **kwargs: piped to plot_timeheight2 function

    Returns:
        ``fig, ax``
    """
    data_cont['colormap'] = plt.cm.get_cmap('jet', 7)
    # We must be sure to specify the ticks matching our target names

    data_cont['mask'] = (data_cont['var'] < 0)
    data_cont["var_lims"] = [-0.5, 6.5]
    fig, ax = Transf.plot_timeheight2(data_cont, **kwargs)
    #ax.cbar.set_yticks([0, 1, 2, 3, 4, 5])
    cbar = fig.axes[1]
    #cbar.set_ylabel("Number of nodes", fontweight='semibold', fontsize=15)
    #print(fig.axes[1].get_yticks())
    #print(cbar.ax.get_ticklocs())
    #print(fig.axes)
    #print(fig.axes[1])
    #fig.axes[1].set_yticklabels(labels.values())
    return fig, ax



def child_iter(indices):
    def children(i):
        "helper to geht the correct order"
        if i in indices:
            yield i
            yield from children(2*i+1)
            yield from children(2*i+2)
    return children



def to_text(data_cont):
    """represent the tree as a multiline string"""
    maxind_at_level = np.cumsum([2**n for n in range(5)]) - 1

    single_tree = data_cont['var']
    no_levels = np.searchsorted(maxind_at_level, max(single_tree.keys()))
    print("no levels", no_levels)

    dt = h.ts_to_dt(data_cont['ts']) 
    lines = []
    lines.append("tree at ts {} and rg {:.2f}".format(
        dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], data_cont['rg']))
    header = "id (bounds)  " + (no_levels)*"   " + \
       "        Z       v      σ      𝛾"
    if single_tree[0]['ldr'] != -99:
        header += "    ldr ldrmax"
    header += "  thres   prom"
    lines.append(header)
    
    # iterate over children depth first, to build a proper tree
    for i in child_iter(list(single_tree.keys()))(0):
        moms = single_tree[i]
        #print(i, moms["bounds"])
        level = np.searchsorted(maxind_at_level, i)
        spc_bef = ""
        if level > 0:
            if np.floor((i-1)/2.)%2 or np.floor((i-1)/2.) == 0:
                if i%2:
                    spc_bef = " " + (level-1)*"|  " + "+-"
                else:
                    spc_bef = " " + (level-1)*"|  " + "`-"
            else:
                if i%2:
                    spc_bef = " " + (level-2)*"|  " + "   +-"
                else:
                    spc_bef = " " + (level-2)*"|  " + "   `-"
        spc_aft = (no_levels-level)*"   "
        bounds_f = '({:>3d}, {:>3d})'.format(*moms['bounds'])

        momstr = '{:> 6.2f}, {:> 6.2f}, {:>5.2f}, {:> 3.2f}'.format(
            moms['z'], moms['v'], moms['width'], moms['skew'])
        if moms['ldr'] != -99:
            momstr += ", {:> 5.1f}, {:> 5.1f}".format(moms['ldr'], moms['ldrmax'])
        momstr += ", {:> 5.1f}, {:> 5.1f}".format(moms['thres'], moms['prominence'])
        txt = "{}{:>2d} {}{} | {}".format(spc_bef, i, bounds_f, spc_aft, momstr)
        lines.append(txt)
    return '\n'.join(lines)


def print_tree(data_cont):
    """print the tree as a multi line string"""
    print(to_text(data_cont))
