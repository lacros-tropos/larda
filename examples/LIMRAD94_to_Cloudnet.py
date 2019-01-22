"""
This routine generates a daily NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
The generated files will be used as input for the Cloudnet processing chain.

Args:
    **date (string): format YYYYMMDD
    **path (string): path where NetCDF file will be stored

Example:
    python LIMRAD94_to_Cloudnet.py date=20181201

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
    Y, M, D = int(date[0:4]), int(date[4:6]), int(date[6:8])
    begin_dt = datetime.datetime(Y, M, D, 0, 0, 0)
    end_dt = datetime.datetime(Y, M, D, 23, 59, 59)
else:
    today = datetime.datetime.now()
    begin_dt = datetime.datetime(today.year, today.month, today.day, 0, 0, 0)
    end_dt = datetime.datetime(today.year, today.month, today.day, 2, 0, 0)

path = kwargs['path'] if 'path' in kwargs else ''

print('    Input values:')
print('         - date:   ', f'{begin_dt:%Y%m%d  from: %H:%M:%S (UTC) to: }{end_dt:%H:%M:%S}', ' (UTC)')

variable_list = ['Ze', 'VEL', 'sw', 'skew', 'kurt', 'DiffAtt', 'ldr', 'bt', 'rr', 'LWP',
                 'MaxVel', 'C1Range', 'C2Range', 'C3Range']

LIMRAD94_vars = {}

for var in variable_list:
    LIMRAD94_vars.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})


flag = nc.generate_cloudnet_input_LIMRAD94(LIMRAD94_vars, path)
