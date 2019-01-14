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

print('days with data', larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6))

#begin_dt=datetime.datetime(2018,12,2,16,0,0)
#end_dt=datetime.datetime(2018,12,2,22,0,0)

plot_range = [300, 10000]

begin_dt=datetime.datetime(2019,1,10,15,0)
end_dt=datetime.datetime(2019,1,10,20,0,0)

#time_of_interest=datetime.datetime(2019,1,10,15,20)
#height_of_interest=3300


MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,'max'])
#fig, ax = pyLARDA.Transformations.plot2d(params["MIRA_Zg"],begin,end,plot_base,plot_top,savename="example/mira_zg.png")
# customly overwrite mira plot limits
MIRA_Zg['var_lims'] = [-40,20]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig('mira_z3.png', dpi=250)

LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig('limrad_Z3.png', dpi=250)


#LIMRAD_Zspec=larda.read("LIMRAD94","C2Hspec",[begin_dt,end_dt],[0,'max'])
#LIMRAD_Zspec=larda.read("LIMRAD94","C3Hspec",[begin_dt,end_dt],[0,'max'])
#print(LIMRAD_Zspec)
#time_list = LIMRAD94_Z['ts']
#dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]
#range_list = LIMRAD94_Z['rg']

#closest_time_limrad=min(dt_list,key=lambda x: abs(x-time_of_interest))
#tind_limrad=dt_list.index(closest_time_limrad)
#closest_height_limrad = min(range_list, key=lambda x: abs(x - height_of_interest))
#hind_limrad=range_list.index(closest_height_limrad)

# look for snippets in Plot_Doppler_Spectra_LIMRad_MIRA