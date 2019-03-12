

cimport cython
import numpy as np


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def build_tree(double[:, :] data, bint ldr_avail):
    """
    'parent' #0
    'Z' #1
    'v' #2
    'width' #3
    'skew' #4
    'threshold' #5
    'prominence' #6
    'bound_l' #7
    'bound_r' #8
    'LDR' #9
    'ldrmax' #10
    """
    cdef dict travtree = {}    
    cdef dict node
    cdef int k
    
    #logger.debug('peakTree parent {}'.format(ncD.variables['parent'][it,ir,:]))
    avail_nodes = np.argwhere(np.asarray(data[:,0]) != -999.).ravel()
    #print("within build_tree", avail_nodes)
    for k in np.asarray(avail_nodes).astype(int).tolist():
        node = {'parent_id': data[k,0], 
                'thres': data[k,5],
                'width': data[k,3], 
                'z': data[k,1], 
                'bounds': (data[k,7], data[k,8]),
                'skew': data[k,4],
                'prominence': data[k,6],
                'v': data[k,2]}
        #print("done build node, now configuring")
        node['id'] = k
        node['bounds'] = list(map(int, node['bounds']))
        node['width'] = node['width'] if np.isfinite(node['width']) else -99
        node['skew'] = node['skew'] if np.isfinite(node['skew']) else -99
        node['thres'] = node['thres'] if np.isfinite(node['thres']) else -99
        node['prominence'] = node['prominence'] if np.isfinite(node['prominence']) else -99
        if ldr_avail:
            node['ldr'] = data[k,9]
            node['ldr'] = node['ldr'] if np.isfinite(node['ldr']) else -99
            node['ldrmax'] = data[k,10]
            node['ldrmax'] = node['ldrmax'] if np.isfinite(node['ldrmax']) else -99
        else:
            node['ldr'], node['ldrmax'] = -99, -99
        #print("done build node, now configuring")
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

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def array_to_tree_c(double[:, :, :, :] data, bint ldr_avail):
    trees = np.empty((data.shape[0], data.shape[1]), dtype=object)
    mask = np.empty((data.shape[0], data.shape[1]), dtype=bool)
    mask[:] = True
    cdef double[:,:,:,:] data_view = data
    cdef int it, ir
    
    for it in range(data.shape[0]):
        #print(it)
        for ir in range(data.shape[1]):
            #print(np.asarray(data[it,ir,:,0]))
            trees[it, ir] = build_tree(data[it,ir,:,:], ldr_avail)
            mask[it, ir] = False
            
    return trees, mask