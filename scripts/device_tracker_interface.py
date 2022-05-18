#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('../')

import pyLARDA
import pyLARDA.helpers as h
import datetime

import pprint

import logging
log = logging.getLogger('pyLARDA')
#log.setLevel(logging.DEBUG)
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

#Load LARDA
larda = pyLARDA.LARDA('local')

instr_filter = ['instr_name', 'hatpro_lacros_g2']
build_lists = True


aggregate = {}

for c in larda.campaign_list:
    larda.connect(c, build_lists=build_lists, filt=instr_filter)
    conn = larda.connectors
    if len(conn) > 1:
        raise ValueError(f'filter {filt} matches more than once')
    elif len(conn) == 0:
        continue

    ckey, cvalue = conn.popitem()
    # take all keys from params config, that are optional (template, instr_name, also_match... )
    free_form_keys = {k:v for k,v in cvalue.system_info.items() if not k in ['path', 'generic', 'params']}
    aggregate[c] = {
        'meta': free_form_keys,
        'camp_info': larda.camp.info_dict,
        'valid_dates': cvalue.valid_dates,
        'system_name': ckey, # system name in that campaign (might change, depending on config)
        'files': cvalue.filehandler,
        'path_pattern': cvalue.system_info['path']
    }
    cvalue.system_info


#pprint.pprint(aggregate)
print('\n\n')
print(f"instruemnt {instr_filter[0]}:{instr_filter[1]} took place in:")
print('\n')
for k, v in aggregate.items():
    print(f"---   {k}    duration {v['valid_dates']}  configured as {v['system_name']}")
    pprint.pprint(v['camp_info'])
    pprint.pprint(v['meta'])
    print('with the path_patterns:')
    pprint.pprint(v['path_pattern'])
    print('    ')



# ----------------------------------------------
# example for optaining the filenames 
# ----------------------------------------------

filter_between = ['20220101-000000', '20220504-000000']
filter_ncpath = []
filter_ncpath = ['BRTbin']
filter_ncpath = ['BRTbin', 'LWPbin']


def check_within(filedates, filterdates):
    return (filedates[0] >= filterdates[0]) and (filedates[1] <= filterdates[1])

selected_files = []
for k,v in aggregate.items():

    for p in set(v['path_pattern']) & set(filter_ncpath):
        base_dir = v['path_pattern'][p]['base_dir'] 
        files = [base_dir + f[1] for f in v['files'][p] if check_within(f[0], filter_between)]
        selected_files += files

print('files that matched the criteria ', len(selected_files))
print(selected_files)


