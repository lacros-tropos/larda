#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('../')
sys.path.append('.')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np
import scipy.ndimage as spn
from scipy import stats

#Load LARDA
#larda=pyLARDA.LARDA('lacros_dacapo')

larda=pyLARDA.LARDA().connect_local('lacros_dacapo')
#c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6)) 


#begin_dt=datetime.datetime(2018,12,6,0,1)
begin_dt=datetime.datetime(2018,12,6,1,40)
end_dt=datetime.datetime(2018,12,6,4,0,0)

plot_range = [300, 10000]
case_prefix = 'plots/scatter_case_studies/comparison_cloud_'

# load the reflectivity data
MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,'max'])
MIRA_Zg['var_lims'] = [-40,20]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mira_z.png', dpi=250)


LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z.png', dpi=250)

LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z_interp.png', dpi=250)

# Z scatterplot

combined_mask = np.logical_or(MIRA_Zg['mask'], LIMRAD94_Z_interp['mask'])

Mira_Z_scatter = h.lin2z(MIRA_Zg['var'][~combined_mask].ravel())#+4.5
Limrad_Z_scatter = h.lin2z(LIMRAD94_Z_interp['var'][~combined_mask].ravel())

s, i, r, p, std_err = stats.linregress(Mira_Z_scatter, Limrad_Z_scatter)
H, xedges, yedges = np.histogram2d(Mira_Z_scatter, Limrad_Z_scatter, 
                                   bins=120, range=[[-75, 30], [-75, 30]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)

# helper lines (1:1), ...
ax.plot([-80,30],[-80,30], color='salmon')
ax.plot([-80,30],[-90,20], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-85,25], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-75,35], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-70,40], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-75,20])
ax.set_ylim([-75,20])
ax.set_xlabel("MIRA Z")
ax.set_ylabel("LIMRAD94 Z")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_mira_limrad.png', dpi=250)

# correct the found bias
def bias(datalist):
    var = datalist[0]['var'].copy() * h.z2lin(4.7)
    mask = datalist[0]['mask']
    return var, mask
MIRA_Zg = pyLARDA.Transformations.combine(bias, [MIRA_Zg], MIRA_Zg['paraminfo'])
fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mira_z_bias_4-7.png', dpi=250)

# calc and plot the DWR
def calc_DWR(datalist):
    var = h.lin2z(datalist[0]['var']) - h.lin2z(datalist[1]['var'])
    mask = np.logical_or(datalist[0]['mask'], datalist[1]['mask'])
    return var, mask
pinfo = {'system': '', 'name': 'DWR', 'colormap': 'gist_rainbow', 'rg_unit': 'm', 'var_lims': [-3, 7]}
DWR = pyLARDA.Transformations.combine(calc_DWR, [MIRA_Zg, LIMRAD94_Z_interp], pinfo)
fig, ax = pyLARDA.Transformations.plot2d(DWR, range_interval=plot_range)
fig.savefig(case_prefix+'DWR_MIRA_LIMRAD.png', dpi=250)

# other day for the mrr

begin_dt=datetime.datetime(2018,12,9,2,0)
end_dt=datetime.datetime(2018,12,9,5,0,0)
case_prefix = 'plots/scatter_case_studies/comparison_rain_'

MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,4000])
MIRA_Zg['var_lims'] = [-40,20]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mira_z.png', dpi=250)


LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,4000])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z.png', dpi=250)
LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z_interp.png', dpi=250)

combined_mask = np.logical_or(MIRA_Zg['mask'], LIMRAD94_Z_interp['mask'])

Mira_Z_scatter = h.lin2z(MIRA_Zg['var'][~combined_mask].ravel())#+4.5
Limrad_Z_scatter = h.lin2z(LIMRAD94_Z_interp['var'][~combined_mask].ravel())

s, i, r, p, std_err = stats.linregress(Mira_Z_scatter, Limrad_Z_scatter)
H, xedges, yedges = np.histogram2d(Mira_Z_scatter, Limrad_Z_scatter, 
                                   bins=120, range=[[-75, 30], [-75, 30]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-80,30],[-80,30], color='salmon')
ax.plot([-80,30],[-90,20], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-85,25], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-75,35], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-70,40], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-75,20])
ax.set_ylim([-75,20])
ax.set_xlabel("MIRA Z")
ax.set_ylabel("LIMRAD94 Z")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_mira_limrad.png', dpi=250)


MRR_Z=larda.read("MRRPRO","Ze",[begin_dt,end_dt],[0,4000])
fig, ax = pyLARDA.Transformations.plot2d(MRR_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mrr_Z.png', dpi=250)
MRR_Z_interp = pyLARDA.Transformations.interpolate2d(MRR_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(MRR_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mrr_Z_interp.png', dpi=250)

# scatterplot
combined_mask = np.logical_or(MIRA_Zg['mask'], MRR_Z_interp['mask'])

Mira_Z_scatter = h.lin2z(MIRA_Zg['var'][~combined_mask].ravel())#+4.5
Mrr_Z_scatter = h.lin2z(MRR_Z_interp['var'][~combined_mask].ravel())

s, i, r, p, std_err = stats.linregress(Mira_Z_scatter, Mrr_Z_scatter)
H, xedges, yedges = np.histogram2d(Mira_Z_scatter, Mrr_Z_scatter, 
                                   bins=120, range=[[-75, 30], [-75, 30]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-80,30],[-80,30], color='salmon')
ax.plot([-80,30],[-90,20], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-85,25], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-75,35], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-70,40], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-75,20])
ax.set_ylim([-75,20])
ax.set_xlabel("MIRA Z")
ax.set_ylabel("MRR Z")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_mira_MRR.png', dpi=250)


MRR_Z_interp = pyLARDA.Transformations.interpolate2d(MRR_Z, new_time=LIMRAD94_Z['ts'], new_range=LIMRAD94_Z['rg'])
fig, ax = pyLARDA.Transformations.plot2d(MRR_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mrr_Z_interp_to_limrad.png', dpi=250)

combined_mask = np.logical_or(LIMRAD94_Z['mask'], MRR_Z_interp['mask'])

Limrad_Z_scatter = h.lin2z(LIMRAD94_Z['var'][~combined_mask].ravel())#+4.5
Mrr_Z_scatter = h.lin2z(MRR_Z_interp['var'][~combined_mask].ravel())

s, i, r, p, std_err = stats.linregress(Limrad_Z_scatter, Mrr_Z_scatter)
H, xedges, yedges = np.histogram2d(Limrad_Z_scatter, Mrr_Z_scatter, 
                                   bins=120, range=[[-75, 30], [-75, 30]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-80,30],[-80,30], color='salmon')
ax.plot([-80,30],[-90,20], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-85,25], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-75,35], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-70,40], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-75,20])
ax.set_ylim([-75,20])
ax.set_xlabel("LIMRAD Z")
ax.set_ylabel("MRR Z")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_limrad_MRR.png', dpi=250)



#begin_dt=datetime.datetime(2018,12,6,0,1)
begin_dt=datetime.datetime(2018,12,14,6,0)
end_dt=datetime.datetime(2018,12,14,9,0)

plot_range = [300, 10000]
case_prefix = 'plots/scatter_case_studies/comparison_c14_'

# load the reflectivity data
MIRA_Zg=larda.read("MIRA","Zg",[begin_dt,end_dt],[0,'max'])
MIRA_Zg['var_lims'] = [-40,20]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'mira_z.png', dpi=250)


LIMRAD94_Z=larda.read("LIMRAD94","Ze",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z.png', dpi=250)

LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, new_time=MIRA_Zg['ts'], new_range=MIRA_Zg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_Z_interp, range_interval=plot_range, z_converter='lin2z')
fig.savefig(case_prefix+'limrad_Z_interp.png', dpi=250)

# Z scatterplot

combined_mask = np.logical_or(MIRA_Zg['mask'], LIMRAD94_Z_interp['mask'])

Mira_Z_scatter = h.lin2z(MIRA_Zg['var'][~combined_mask].ravel())#+4.5
Limrad_Z_scatter = h.lin2z(LIMRAD94_Z_interp['var'][~combined_mask].ravel())

s, i, r, p, std_err = stats.linregress(Mira_Z_scatter, Limrad_Z_scatter)
H, xedges, yedges = np.histogram2d(Mira_Z_scatter, Limrad_Z_scatter, 
                                   bins=120, range=[[-75, 30], [-75, 30]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-80,30],[-80,30], color='salmon')
ax.plot([-80,30],[-90,20], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-85,25], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-75,35], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-80,30],[-70,40], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-75,20])
ax.set_ylim([-75,20])
ax.set_xlabel("MIRA Z")
ax.set_ylabel("LIMRAD94 Z")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_mira_limrad.png', dpi=250)
