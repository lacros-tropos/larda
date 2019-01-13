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
larda = pyLARDA.LARDA().connect('lacros_dacapo')
c_info = [larda.camp.LOCATION, larda.camp.VALID_DATES]

print(larda.days_with_data())
#print("array_avail()", larda.array_avail(2015, 6))
#print("single month with new interface ", larda.instr_status(2015, 6)) 


#begin_dt=datetime.datetime(2018,12,6,0,1)
begin_dt=datetime.datetime(2018,12,6,1,40)
end_dt=datetime.datetime(2018,12,6,4,0,0)

plot_range = [300, 10000]
case_prefix = 'plots/scatter_case_studies/comparison_cloud_'

#### load the velocity data

MIRA_VELg=larda.read("MIRA","VELg",[begin_dt,end_dt],[0,'max'])
MIRA_VELg['var_lims'] = [-6,6]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_VELg, range_interval=plot_range)
fig.savefig(case_prefix+'mira_vel.png', dpi=250)

LIMRAD94_VEL=larda.read("LIMRAD94","VEL",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel.png', dpi=250)
LIMRAD94_VEL_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL, new_time=MIRA_VELg['ts'], new_range=MIRA_VELg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL_interp, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel_interp.png', dpi=250)

# vel scatterplot
combined_mask = np.logical_or(MIRA_VELg['mask'], LIMRAD94_VEL_interp['mask'])

Mira_VELg_scatter = MIRA_VELg['var'][~combined_mask].ravel()
Limrad_VEL_scatter = LIMRAD94_VEL_interp['var'][~combined_mask].ravel()


s, i, r, p, std_err = stats.linregress(Mira_VELg_scatter, Limrad_VEL_scatter)
H, xedges, yedges = np.histogram2d(Mira_VELg_scatter, Limrad_VEL_scatter, 
                                   bins=120, range=[[-4, 4], [-4, 4]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-10,10],[-10,10], color='salmon')
ax.plot([-10,10],[-8,12], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-9,11], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-11,9], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-12,8], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_xlabel("MIRA VELg")
ax.set_ylabel("LIMRAD94 VEL")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_vel_mira_limrad.png', dpi=250)



# other day for the mrr
begin_dt=datetime.datetime(2018,12,9,2,0)
end_dt=datetime.datetime(2018,12,9,5,0,0)
case_prefix = 'plots/scatter_case_studies/comparison_rain_'

MIRA_VELg=larda.read("MIRA","VELg",[begin_dt,end_dt],[0,4000])
MIRA_VELg['var_lims'] = [-6,6]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_VELg, range_interval=plot_range)
fig.savefig(case_prefix+'mira_vel.png', dpi=250)

LIMRAD94_VEL=larda.read("LIMRAD94","VEL",[begin_dt,end_dt],[0,4000])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel.png', dpi=250)
LIMRAD94_VEL_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL, new_time=MIRA_VELg['ts'], new_range=MIRA_VELg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL_interp, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel_interp.png', dpi=250)

# vel scatterplot
combined_mask = np.logical_or(MIRA_VELg['mask'], LIMRAD94_VEL_interp['mask'])

Mira_VELg_scatter = MIRA_VELg['var'][~combined_mask].ravel()
Limrad_VEL_scatter = LIMRAD94_VEL_interp['var'][~combined_mask].ravel()

s, i, r, p, std_err = stats.linregress(Mira_VELg_scatter, Limrad_VEL_scatter)
H, xedges, yedges = np.histogram2d(Mira_VELg_scatter, Limrad_VEL_scatter, 
                                   bins=120, range=[[-4, 4], [-4, 4]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-10,10],[-10,10], color='salmon')
ax.plot([-10,10],[-8,12], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-9,11], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-11,9], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-12,8], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_xlabel("MIRA VELg")
ax.set_ylabel("LIMRAD94 VEL")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_vel_mira_limrad.png', dpi=250)


MRR_VEL=larda.read("MRRPRO","VEL",[begin_dt,end_dt],[0,4000])
fig, ax = pyLARDA.Transformations.plot2d(MRR_VEL, range_interval=plot_range)
fig.savefig(case_prefix+'mrr_vel.png', dpi=250)
MRR_VEL_interp = pyLARDA.Transformations.interpolate2d(MRR_VEL, new_time=MIRA_VELg['ts'], new_range=MIRA_VELg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(MRR_VEL_interp, range_interval=plot_range)
fig.savefig(case_prefix+'mrr_vel_interp.png', dpi=250)

# scatterplot
combined_mask = np.logical_or(MIRA_VELg['mask'], MRR_VEL_interp['mask'])

Mira_VELg_scatter = MIRA_VELg['var'][~combined_mask].ravel()
MRR_VEL_scatter = MRR_VEL_interp['var'][~combined_mask].ravel()

s, i, r, p, std_err = stats.linregress(Mira_VELg_scatter, MRR_VEL_scatter)
H, xedges, yedges = np.histogram2d(Mira_VELg_scatter, MRR_VEL_scatter, 
                                   bins=120, range=[[-4, 4], [-4, 4]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-10,10],[-10,10], color='salmon')
ax.plot([-10,10],[-8,12], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-9,11], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-11,9], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-12,8], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_xlabel("MIRA VELg")
ax.set_ylabel("MRRPRO VEL")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_vel_mira_mrr.png', dpi=250)


MRR_VEL_interp = pyLARDA.Transformations.interpolate2d(MRR_VEL, new_time=LIMRAD94_VEL['ts'], new_range=LIMRAD94_VEL['rg'])
fig, ax = pyLARDA.Transformations.plot2d(MRR_VEL_interp, range_interval=plot_range)
fig.savefig(case_prefix+'mrr_vel_interp_to_limrad.png', dpi=250)

combined_mask = np.logical_or(LIMRAD94_VEL['mask'], MRR_VEL_interp['mask'])

LIMRAD94_VEL_scatter = LIMRAD94_VEL['var'][~combined_mask].ravel()
MRR_VEL_scatter = MRR_VEL_interp['var'][~combined_mask].ravel()

s, i, r, p, std_err = stats.linregress(LIMRAD94_VEL_scatter, MRR_VEL_scatter)
H, xedges, yedges = np.histogram2d(LIMRAD94_VEL_scatter, MRR_VEL_scatter, 
                                   bins=120, range=[[-4, 4], [-4, 4]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-10,10],[-10,10], color='salmon')
ax.plot([-10,10],[-8,12], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-9,11], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-11,9], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-12,8], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_xlabel("LIMRAD VEL")
ax.set_ylabel("MRRPRO VEL")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_vel_limrad_mrr.png', dpi=250)



begin_dt=datetime.datetime(2018,12,14,6,0)
end_dt=datetime.datetime(2018,12,14,9,0)

plot_range = [300, 10000]
case_prefix = 'plots/scatter_case_studies/comparison_c14_'

#### load the velocity data

MIRA_VELg=larda.read("MIRA","VELg",[begin_dt,end_dt],[0,'max'])
MIRA_VELg['var_lims'] = [-6,6]
fig, ax = pyLARDA.Transformations.plot2d(MIRA_VELg, range_interval=plot_range)
fig.savefig(case_prefix+'mira_vel.png', dpi=250)

LIMRAD94_VEL=larda.read("LIMRAD94","VEL",[begin_dt,end_dt],[0,'max'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel.png', dpi=250)
LIMRAD94_VEL_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_VEL, new_time=MIRA_VELg['ts'], new_range=MIRA_VELg['rg'])
fig, ax = pyLARDA.Transformations.plot2d(LIMRAD94_VEL_interp, range_interval=plot_range)
fig.savefig(case_prefix+'limrad_vel_interp.png', dpi=250)

# vel scatterplot
combined_mask = np.logical_or(MIRA_VELg['mask'], LIMRAD94_VEL_interp['mask'])

Mira_VELg_scatter = MIRA_VELg['var'][~combined_mask].ravel()
Limrad_VEL_scatter = LIMRAD94_VEL_interp['var'][~combined_mask].ravel()


s, i, r, p, std_err = stats.linregress(Mira_VELg_scatter, Limrad_VEL_scatter)
H, xedges, yedges = np.histogram2d(Mira_VELg_scatter, Limrad_VEL_scatter, 
                                   bins=120, range=[[-4, 4], [-4, 4]])
X, Y = np.meshgrid(xedges,yedges)
fig, ax = plt.subplots(1, figsize=(5.7, 5.7))
ax.pcolormesh(X, Y, np.transpose(H),
              norm=matplotlib.colors.LogNorm(),
)
ax.text(0.01, 0.93, 'slope {:5.3f}\nintercept {:5.3f}\nR^2 {:5.3f}'.format(s,i,r**2), 
        horizontalalignment='left',
        verticalalignment='center', transform=ax.transAxes)
#ax.scatter(Mira_Z_scatter, Limrad_Z_scatter, s=1)
ax.plot([-10,10],[-10,10], color='salmon')
ax.plot([-10,10],[-8,12], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-9,11], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-11,9], color='salmon', linewidth=0.7, linestyle='--')
ax.plot([-10,10],[-12,8], color='salmon', linewidth=0.7, linestyle='--')
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_xlabel("MIRA VELg")
ax.set_ylabel("LIMRAD94 VEL")
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.tick_params(axis='both', which='both', right=True, top=True)
fig.savefig(case_prefix+'scatter_vel_mira_limrad.png', dpi=250)
