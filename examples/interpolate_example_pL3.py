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
import numpy as np
import scipy.ndimage as spn

import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

#Load LARDA
larda=pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6)) 


begin_dt=datetime.datetime(2018,12,6,0,1)
end_dt=datetime.datetime(2018,12,6,4,0,0)

plot_range = [300, 10000]

MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,'max'])
MIRA_Zg['var_lims'] = [-40,20]
#fig, ax = pyLARDA.Transformations.plot_timeheight(params["MIRA_Zg"],begin,end,plot_base,plot_top,savename="example/mira_zg.png")
fig, ax = pyLARDA.Transformations.plot_timeheight(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig('mira_z.png', dpi=250)


LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig('limrad_Z.png', dpi=250)
#with np.printoptions(precision=12):
#    print("timestamps mira", MIRA_Zg['ts'].shape, MIRA_Zg['ts'].astype(float))
#    print("timesdiff mira", np.diff(MIRA_Zg['ts']))
#    print("timestamps limrad", MIRA_Zg['ts'].shape, LIMRAD94_Z['ts'].astype(float))
#    print("timesdiff limrad", np.diff(LIMRAD94_Z['ts']))
LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])
fig, ax = pyLARDA.Transformations.plot_timeheight(LIMRAD94_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig('limrad_Z_interp.png', dpi=250)

def calc_DWR(datalist):
    var = h.lin2z(datalist[0]['var']) - h.lin2z(datalist[1]['var'])
    mask = np.logical_or(datalist[0]['mask'], datalist[1]['mask'])
    return var, mask

new_keys = {'system': '', 'name': 'DWR', 'colormap': 'gist_rainbow', 'rg_unit': 'm', 'var_lims': [-7, 2]}
DWR = pyLARDA.Transformations.combine(calc_DWR, [MIRA_Zg, LIMRAD94_Z_interp], new_keys)
fig, ax = pyLARDA.Transformations.plot_timeheight(DWR, range_interval=plot_range)
fig.savefig('DWR_MIRA_LIMRAD.png', dpi=250)

MRR_Z=larda.read("MRRPRO","Ze",[begin_dt,end_dt],[0,'max'])
MRR_Z['var'] = h.lin2z(MRR_Z['var'])
fig, ax = pyLARDA.Transformations.plot_timeheight(MRR_Z, range_interval=plot_range)
fig.savefig('mrr_Z.png', dpi=250)

