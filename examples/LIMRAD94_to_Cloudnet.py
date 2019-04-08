"""
This routine generates a daily NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
The generated files will be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python LIMRAD94_to_Cloudnet.py date=20181201 path=/tmp/pycharm_project_626/scripts_Willi/cloudnet_input/

"""


import sys
sys.path.append('../')
sys.path.append('.')

import pyLARDA
import pyLARDA.NcWrite as nc
import pyLARDA.helpers as h

import datetime
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print('available systems:', larda.connectors.keys())
print("available parameters: ", [(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())

# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)

# gather argument
if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = datetime.datetime.strptime(date+' 00:00:05', '%Y%m%d %H:%M:%S')
    end_dt   = datetime.datetime.strptime(date+' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    #today = datetime.datetime.now()
    #begin_dt = datetime.datetime(today.year, today.month, today.day, 0)
    #end_dt = datetime.datetime(today.year, today.month, today.day, 2)
    date = '20190327'
    begin_dt = datetime.datetime.strptime(date+' 0:00:05', '%Y%m%d %H:%M:%S')
    end_dt   = datetime.datetime.strptime(date+' 23:59:55', '%Y%m%d %H:%M:%S')


cloudnet_remsens_lim_path = '/lacroshome/remsens_lim/data/cloudnet/'

if 'path' in kwargs:
    path = kwargs['path']
else:
    if c_info[0] == 'Punta Arenas':
        path = cloudnet_remsens_lim_path + 'punta-arenas/' + 'calibrated/limrad94/' + date[:4] + '/'
    elif c_info[0] == 'Leipzig':
        path = cloudnet_remsens_lim_path + 'leipzig/' + 'calibrated/limrad94/' + date[:4] + '/'
    else:
        print('Error: No other sites implemented jet!')
        sys.exit(-42)


print('         Input date: {}  from: {} (UTC) to: {} (UTC)\n'.format(begin_dt.strftime("%Y%m%d"),
                                                                      begin_dt.strftime("%H:%M:%S"),
                                                                      end_dt.strftime("%H:%M:%S")))

variable_list = ['Ze', 'VEL', 'sw', 'skew', 'kurt', 'DiffAtt', 'ldr', 'bt', 'rr', 'LWP',
                 'MaxVel', 'C1Range', 'C2Range', 'C3Range']

LIMRAD94_vars = {}

for var in variable_list:
    print('variable :: ' + var)
    #if var in ["Ze", "VEL", 'sw', "skew", 'kurt', 'DiffAtt', 'ldr']:
    #    kwargs = {'interp_rg_join': True}
    #else:
    kwargs = {}
    LIMRAD94_vars.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'], **kwargs)})


flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_vars, path)
