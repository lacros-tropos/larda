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


print('available systems:', larda.connectors.keys())
print("available parameters: ",[(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
print('days with data', larda.days_with_data())


print(larda.description("MIRA","Zg"))

print(larda.description("LIMRAD94","Ze"))