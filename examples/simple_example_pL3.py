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

import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

#Load LARDA
larda=pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print('available systems:', larda.connectors.keys())
print("available parameters: ",[(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6)) 

#begin_dt=datetime.datetime(2018,12,2,16,0,0)
#end_dt=datetime.datetime(2018,12,2,22,0,0)

begin_dt=datetime.datetime(2018,12,18,6,0)
end_dt=datetime.datetime(2018,12,18,11,0,0)
plot_range = [5500, 7500]


MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot_timeheight(params["MIRA_Zg"],begin,end,plot_base,plot_top,savename="example/mira_zg.png")
# customly overwrite mira plot limits
MIRA_Zg['var_lims'] = [-40,20]
fig, ax = pyLARDA.Transformations.plot_timeheight(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig('mira_z.png', dpi=250)
h.pprint(MIRA_Zg)

LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig('limrad_Z.png', dpi=250)

MRR_Z=larda.read("MRRPRO","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(MRR_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig('mrr_Z.png', dpi=250)

shaun_vel=larda.read("SHAUN","VEL",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(shaun_vel, range_interval=plot_range)
fig.savefig('shaun_vel.png', dpi=250)


cloudnet_beta=larda.read("CLOUDNET","beta",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(cloudnet_beta, range_interval=plot_range, z_converter="log")
fig.savefig('cloudnet_beta.png', dpi=250)

cloudnet_depol=larda.read("CLOUDNET","depol",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(cloudnet_depol, range_interval=plot_range)
fig.savefig('cloudnet_delta.png', dpi=250)

CLOUDNET_lwp=larda.read("CLOUDNET", "LWP", [begin_dt,end_dt])
fig, ax = pyLARDA.Transformations.plot_timeseries(CLOUDNET_lwp)
fig.savefig('cloudnet_lwp.png', dpi=250)

CLOUDNET_Z=larda.read("CLOUDNET","Z",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(CLOUDNET_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig('cloudnet_Z.png', dpi=250)

CLOUDNET_VEL=larda.read("CLOUDNET","VEL",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(CLOUDNET_VEL, range_interval=plot_range)
fig.savefig('cloudnet_VEL.png', dpi=250)
print("Simple Example - END")
