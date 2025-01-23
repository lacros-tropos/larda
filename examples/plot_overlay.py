
import matplotlib
import datetime
import pyLARDA
import numpy as np
larda=pyLARDA.LARDA()
larda = pyLARDA.LARDA('remote', uri='http://larda3.tropos.de/')
larda.connect('cloudlab_III')

begin_dt=datetime.datetime(2024,1,8,0,0)
begin_dt2=datetime.datetime(2024,1,7,23,0)
end_dt=datetime.datetime(2024,1,8,16,0)
plot_range = [50, 10000]

T = larda.read("CLOUDNET","T", [begin_dt2,end_dt], plot_range)
def toC(datalist):
        return datalist[0]['var']-273.15, datalist[0]['mask']
contour = {
        'data': pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"}),
        'levels': np.arange(-40,11,5)
    }

MIRA_Zg = larda.read("MIRA_MBR7","Zg",[begin_dt,end_dt],[0,'max'])
MIRA_Zg['colormap'] = 'jet'
fig, ax = pyLARDA.Transformations.plot_timeheight2(MIRA_Zg, range_interval=plot_range, fig_size=[20,5.7], z_converter="lin2z",contour=contour)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,2,4,6,8,10,12,14,16]))
ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,]))
fig.savefig('MIRA_Z.png')
