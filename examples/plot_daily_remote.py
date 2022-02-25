#!/usr/bin/env python3
"""


Example usage:

python3 plot_daily_remote.py --campaign lacros_dacapo --system MRRPRO --date YYYYMMDD
python3 plot_daily_remote.py --campaign lacros_dacapo --system MIRA --date YYYYMMDD
python3 plot_daily_remote.py --campaign lacros_dacapo --system LIMRAD94 --date YYYYMMDD

"""

import sys
sys.path.append('../larda/')
import datetime
import argparse 
from pathlib import Path

import pyLARDA
import pyLARDA.Transformations as pLTrans

parser = argparse.ArgumentParser(description='Plot 24h radar observations')
parser.add_argument('--campaign', help='larda campaign name', required=True)
parser.add_argument('--system', help='system to plot', required=True)
parser.add_argument('--date', help='date in the format YYYYMMDD')
args = parser.parse_args()

#spath = Path(f'../plots/{args.campaign}')
#spath.mkdir(exist_ok=True)

dt_begin = datetime.datetime.strptime(args.date, '%Y%m%d')
dt_end = dt_begin + datetime.timedelta(hours=24) 
print(f"time range {dt_begin}  -  {dt_end}")


# ----------------------------------------------------
# add the uri of your remote here 
larda = pyLARDA.LARDA('remote', uri='')
larda.connect(args.campaign, build_lists=False)

# link the param names in larda, to the ones expected by the
# plotting function (necessary as remsens_limrad_quicklooks was designed for rpg94)
param_names = {
    "MIRA": {"Ze": "MIRA|Ze", "VEL": "MIRA|VELg", "sw": "MIRA|sw",
             "ldr": "MIRA|LDRg", "LWP": "CLOUDNET|LWP", "rr": "CLOUDNET|rainrate",},
    "LIMRAD94": {"Ze": "LIMRAD94|Ze", "VEL": "LIMRAD94|VEL", "sw": "LIMRAD94|sw",
                 "ldr": "LIMRAD94|ldr", "ZDR": "LIMRAD94|ZDR", "RHV": "LIMRAD94|RHV",
                 "LWP": "LIMRAD94|LWP", "rr": "LIMRAD94|rr",
                 "SurfTemp": "LIMRAD94|SurfTemp", "SurfWS": "LIMRAD94|SurfWS"},
    "RPG94_LACROS": {"Ze": "RPG94_LACROS|Ze", "VEL": "RPG94_LACROS|VEL", "sw": "RPG94_LACROS|sw",
                 "ldr": "RPG94_LACROS|ldr", "ZDR": "RPG94_LACROS|ZDR", "RHV": "RPG94_LACROS|RHV",
                 "LWP": "RPG94_LACROS|LWP", "rr": "RPG94_LACROS|rr",
                 "SurfTemp": "RPG94_LACROS|SurfTemp", "SurfWS": "RPG94_LACROS|SurfWS"},
    "MRRPRO": {"Ze": "MRRPRO|Ze", "VEL": "MRRPRO|VEL", "sw": "MRRPRO|sw",
               "LWP": "CLOUDNET|LWP", "rr": "CLOUDNET|rainrate",},
}


container = {}

for k, v in param_names[args.system].items():
    print(k, v)
    sys, param = v.split("|")
    container[k] = larda.read(sys, param, [dt_begin, dt_end], [0, 12000])

fig, ax = pLTrans.remsens_limrad_quicklooks(container, plot_range=[0, 12000], timespan='24h', mask_jumps=True)
sname = f"{dt_begin.strftime('%Y%m%d')}_{args.campaign}_{args.system}_moments.png"
print('savename ', sname)
fig.savefig(sname)

if 'ZDR' in container:
    fig, ax = pLTrans.remsens_limrad_polarimetry_quicklooks(container, plot_range=[0, 12000], timespan='24h')
    sname = f"{dt_begin.strftime('%Y%m%d')}_{args.campaign}_{args.system}_polarimetry.png"
    print('savename ', sname)
    fig.savefig(sname)
