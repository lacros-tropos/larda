#!
# Copyright (c) 2017 Siphon Contributors.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Wyoming Upper Air Data Request
==============================

This routine gathers sounding data from the Wyoming upper air archive and generates two text files containing sounding
data and meta data, also a png of the sounding is provided.

This example shows how to use siphon's `simplewebswervice` support to create a query to
the Wyoming upper air archive.


necessary packages:

       conda install -c conda-forge metpy
       conda install -c conda-forge siphon


Args:
    **date (integer): format YYYYMMDD
    **hour (integer): e.g. 12
    **sation (string): station identifier, e.g. Punta-Arenas: SCCI
    **folder (string): path to folder where output is saved, default save where script is executed

Return:
    [date]_[hour].png:          plot of sounding
    [date]_[hour]_metadata.txt: meta data of sounding
    [date]_[hour]_sounding.txt: actual data of sounding


Example:
    python download_plot_sounding.py date=20181214 hour=12 station=SCCI

"""

import sys
sys.path.append('../')
sys.path.append('.')

from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.wyoming as w

####################################################
# Create a datetime object for the sounding and string of the station identifier.
# gather arguments
station = 'SCCI'    # punta arenas chile
station = 'EHDB'    # nehterlands near cabauw, hours possible: 00, 12

method_name, args, kwargs = h._method_info_from_argv(sys.argv)

if 'date' in kwargs:
    date_str = str(kwargs['date'])
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
else:
    year, month, day = 2019, 3, 15

station     = kwargs['station'] if 'station' in kwargs else 'SCII'
hour        = int(kwargs['hour'])    if 'hour'    in kwargs else 12
output_path = kwargs['folder']  if 'folder'  in kwargs else ''

date = datetime(year, month, day, hour)

####################################################
# Make the request (a pandas dataframe is returned).
df = w.WyomingUpperAir.request_data(date, station)

# Drop any rows with all NaN values for T, Td, winds
# df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed',
#                       'u_wind', 'v_wind'), how='all').reset_index(drop=True)

##########################################################################

# We will pull the data out of the example dataset into individual variables and
# assign units.

p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)
u_wind = df['u_wind'].values * units(df.units['u_wind'])
v_wind = df['v_wind'].values * units(df.units['v_wind'])

##########################################################################
# Thermodynamic Calculations
# --------------------------
#
# Often times we will want to calculate some thermodynamic parameters of a
# sounding. The MetPy calc module has many such calculations already implemented!
#
# * **Lifting Condensation Level (LCL)** - The level at which an air parcel's
#   relative humidity becomes 100% when lifted along a dry adiabatic path.
# * **Parcel Path** - Path followed by a hypothetical parcel of air, beginning
#   at the surface temperature/pressure and rising dry adiabatically until
#   reaching the LCL, then rising moist adiabatically.

# Calculate the LCL
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

print(lcl_pressure, lcl_temperature)

# Calculate the parcel profile.
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')

##########################################################################
#  Skew-T Plotting
# ------------------------
#
# Fiducial lines indicating dry adiabats, moist adiabats, and mixing ratio are
# useful when performing further analysis on the Skew-T diagram. Often the
# 0C isotherm is emphasized and areas of CAPE and CIN are shaded.

# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=30)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)
skew.ax.set_xlabel('Temperature [°C]')
skew.ax.set_ylabel('Pressure [hPa]')

# Plot LCL temperature as black dot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Plot the parcel profile as a black line
skew.plot(p, parcel_prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, parcel_prof)
skew.shade_cape(p, T, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Add a title
plt.title(str(date) + ' ' + station + ' Punta Arenas')

# Add Legend
plt.legend(['Temperature', 'Dew Point', 'LCL', 'parcel profile'])

# Save the Figure and the data
filename = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2) + '_' + str(date.hour) + '_' + station

file = output_path + filename + '.png'
fig.savefig(file, dpi=100, format='png')
plt.close()

df.to_csv(output_path + filename + '_sounding' + '.txt', sep='\t', index=None)

with open(output_path + filename + '_metadata' + '.txt', 'w') as f:
    for item in df._metadata:
        for item1, item2 in item.items():
            f.write(str(item1) + '\t' + str(item2) + '\n')

print('    Save File :: ' + file)
print('    Save File :: ' + output_path + filename + '_metadata' + '.txt')
print('    Save File :: ' + output_path + filename + '_sounding' + '.txt')
