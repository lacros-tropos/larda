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

#Load LARDA
larda=pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6)) 

#begin_dt=datetime.datetime(2018,12,2,16,0,0)
#end_dt=datetime.datetime(2018,12,2,22,0,0)
plot_range = [300, 10000]

begin_dt=datetime.datetime(2018,12,14,8,40)
end_dt=datetime.datetime(2018,12,14,9,20,0)
plot_range = [300, 10000]


MIRA_Zspec=larda.read("MIRA","Zspec",[begin_dt,end_dt],[0,'max'])
print(MIRA_Zspec)


LIMRAD_Zspec=larda.read("LIMRAD94","C1Hspec",[begin_dt,end_dt],[0,'max'])
LIMRAD_Zspec=larda.read("LIMRAD94","C2Hspec",[begin_dt,end_dt],[0,'max'])
LIMRAD_Zspec=larda.read("LIMRAD94","C3Hspec",[begin_dt,end_dt],[0,'max'])
print(LIMRAD_Zspec)
