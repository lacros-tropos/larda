#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')

import matplotlib

matplotlib.use('Agg')

import pyLARDA
import datetime

import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('lacros_dacapo')

#case_prefix = 'plots/scatter_case_studies/comparison_cloud_'
case_prefix = 'plots/'
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())

# begin_dt=datetime.datetime(2018,12,6,0,1)
begin_dt = datetime.datetime(2018, 12, 6, 0, 0, 0)
end_dt   = datetime.datetime(2018, 12, 6, 0, 30, 0)

plot_range = [0, 12000]

# load the reflectivity data
MIRA_Zg = larda.read("MIRA", "Zg", [begin_dt, end_dt], [0, 'max'])
MIRA_Zg['var_lims'] = [-60, 20]

LIMRAD94_Z = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], [0, 'max'])
LIMRAD94_Z['var_lims'] = [-60, 20]

LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])

fig, ax = pyLARDA.Transformations.plot_scatter(MIRA_Zg, LIMRAD94_Z_interp, var_lim=[-75, 20],
                                          custom_offset_lines=5.0, z_converter='lin2z')

fig.savefig(case_prefix+'scatter_mira_limrad_Z.png', dpi=250)

# load the Doppler velocity data
MIRA_VELg = larda.read("MIRA", "VELg", [begin_dt, end_dt], [0, 'max'])
LIMRAD94_VEL = larda.read("LIMRAD94", "VEL", [begin_dt, end_dt], [0, 'max'])

LIMRAD94_VEL_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL,
                                                            new_time=MIRA_VELg['ts'], new_range=MIRA_VELg['rg'])

fig, ax = pyLARDA.Transformations.plot_scatter(MIRA_VELg, LIMRAD94_VEL_interp, var_lim=[-6, 4],
                                          custom_offset_lines=1.0)

fig.savefig(case_prefix+'scatter_mira_limrad_VEL.png', dpi=250)


## load the reflectivity data
#MIRA_Zg = larda.read("CLOUDNET", "Z", [begin_dt, end_dt], [0, 'max'])
#MIRA_Zg['var_lims'] = [-60, 20]
#
#fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
#fig.savefig(case_prefix+'cloudnet_mira_Z.png', dpi=250)
#
#LIMRAD94_Z = larda.read("CLOUDNET_LIMRAD", "Z", [begin_dt, end_dt], [0, 'max'])
#LIMRAD94_Z['var_lims'] = [-60, 20]
#
#fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
#fig.savefig(case_prefix+'cloudnet_limrad_Z.png', dpi=250)


#fig, ax = pyLARDA.Transformations.scatter(MIRA_Zg, LIMRAD94_Z, var_lim=[-75, 20],
#                                          custom_offset_lines=5.0, z_converter='lin2z')
#
#fig.savefig(case_prefix+'cloudnet_scatter_mira_limrad_Z.png', dpi=250)
#
## load the Doppler velocity data
#MIRA_VELg = larda.read("CLOUDNET", "v", [begin_dt, end_dt], [0, 'max'])
#LIMRAD94_VEL = larda.read("CLOUDNET_LIMRAD", "v", [begin_dt, end_dt], [0, 'max'])
#
#
#fig, ax = pyLARDA.Transformations.scatter(MIRA_VELg, LIMRAD94_VEL, var_lim=[-6, 4],
#                                          custom_offset_lines=1.0)
#
#fig.savefig(case_prefix+'cloudnet_scatter_mira_limrad_VEL.png', dpi=250)