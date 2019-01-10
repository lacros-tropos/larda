#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import pyLARDA
import pyLARDA.helpers as h
import datetime
import scipy.ndimage as spn

import numpy as np

#Load LARDA
larda=pyLARDA.LARDA().connect_local('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]


print('available systems:', larda.connectors.keys())
print("available parameters: ",[(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6))

#begin_dt=datetime.datetime(2018,12,2,16,0,0)
#end_dt=datetime.datetime(2018,12,2,22,0,0)




begin_dt = datetime.datetime(2018,12,9,0,0,0)
end_dt   = datetime.datetime(2018,12,9,1,0,0)
plot_range = [0, 12000]


MIRA_Velg=larda.read("MIRA","VELg",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot_timeheight(params["MIRA_Zg"],begin,end,plot_base,plot_top,savename="example/mira_zg.png")
# customly overwrite mira plot limits
MIRA_Velg['var_lims'] = [-4,2]
fig, ax = pyLARDA.Transformations.plot_timeheight(MIRA_Velg, range_interval=plot_range)#, z_converter='lin2z')
fig.savefig('mira_VELg.png', dpi=250)

LIMRAD94_VEL=larda.read("LIMRAD94","VEL",[begin_dt,end_dt],[0,'max'])
LIMRAD94_VEL['var_lims'] = [-4,2]
fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, range_interval=plot_range)#, z_converter='lin2z')
fig.savefig('limrad_VEL.png', dpi=250)

#ts_start = max(min(MIRA_Velg['ts']), min(LIMRAD94_VEL['ts']))
#ts_end   = min(max(MIRA_Velg['ts']), max(LIMRAD94_VEL['ts']))
#
#rg_start = max(min(MIRA_Velg['rg']), min(LIMRAD94_VEL['rg']))
#rg_end   = min(max(MIRA_Velg['rg']), max(LIMRAD94_VEL['rg']))
#
#new_grid = {'new_time': np.arange(ts_start, ts_end, 5.0), 'new_range': np.arange(rg_start, rg_end, 30.0)}
#
#
#LIMRAD94_VEL_interpol = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL, mask_thres=0.1, **new_grid)
#MIRA_Velg_interpol    = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL, mask_thres=0.1, **new_grid)



print("Simple Example - END")
#
#cloudnet_beta=larda.read("CLOUDNET","beta",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot_timeheight(cloudnet_beta, range_interval=plot_range, z_converter="log")
#fig.savefig('cloudnet_beta.png', dpi=250)
#
#cloudnet_depol=larda.read("CLOUDNET","depol",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot_timeheight(cloudnet_depol, range_interval=plot_range)
#fig.savefig('cloudnet_delta.png', dpi=250)
#
#CLOUDNET_lwp=larda.read("CLOUDNET", "LWP", [begin_dt,end_dt])
#fig, ax = pyLARDA.Transformations.plot_timeseries(CLOUDNET_lwp)
#fig.savefig('cloudnet_lwp.png', dpi=250)
#
#CLOUDNET_Z=larda.read("CLOUDNET","Z",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot_timeheight(CLOUDNET_Z, range_interval=plot_range, z_converter='lin2z')
#fig.savefig('cloudnet_Z.png', dpi=250)
