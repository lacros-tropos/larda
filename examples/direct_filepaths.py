#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('../')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as Transf
import datetime
import numpy as np


import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


#Load LARDA
#larda = pyLARDA.LARDA('remote', uri='http://larda3.tropos.de').connect('mosaic', build_lists=False)
#larda = pyLARDA.LARDA('local').connect('lacros_leipzig_example')

larda = pyLARDA.LARDA('filepath').connect()

paths = [
    '../../../../localdata/hatpro_raw_lacros_test/22042411.LWP',
    '../../../../localdata/hatpro_raw_lacros_test/22042412.LWP',
    '../../../../localdata/hatpro_raw_lacros_test/22042413.LWP',
]

# either specify a time interval 
time_int = [datetime.datetime(2022,4,24,11,30),
	    datetime.datetime(2022,4,24,12,30)]
# or if given empty the whole period in the files will be plotted
time_int = []

lwp=larda.read("HATPRObinary", "lwp", time_int, paths=paths)
fig, ax = Transf.plot_timeseries2(lwp)
fig.savefig('hatpro_binary_lwp.png')

paths = [
    '../../../../localdata/hatpro_raw_lacros_test/22042411.TPC',
    '../../../../localdata/hatpro_raw_lacros_test/22042412.TPC',
    '../../../../localdata/hatpro_raw_lacros_test/22042413.TPC',
]

tpc=larda.read("HATPRObinary", "TPC", time_int, [0, 'max'], paths=paths)
fig, ax = Transf.plot_timeheight2(tpc, tdel_jumps=7201)
fig.savefig('hatpro_binary_tpc.png')