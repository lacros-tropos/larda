# Copyright (c) 2013-2015 University Corporation for Atmospheric Research/Unidata.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
"""Read upper air data from the Wyoming archives."""

#!/usr/bin/python3

from io import StringIO
import warnings
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from siphon._tools import get_wind_components
from siphon.http_util import HTTPEndPoint
import datetime
import metpy.units as units
import pyLARDA.helpers as h

warnings.filterwarnings('ignore', 'Pandas doesn\'t allow columns to be created', UserWarning)

class WyomingUpperAir(HTTPEndPoint):
    """Download and parse data from the University of Wyoming's upper air archive."""

    def __init__(self):
        """Set up endpoint."""
        super(WyomingUpperAir, self).__init__('http://weather.uwyo.edu/cgi-bin/sounding')

    @classmethod
    def request_data(cls, time, site_id, **kwargs):
        r"""Retrieve upper air observations from the Wyoming archive.

        Parameters
        ----------
        time : datetime
            The date and time of the desired observation.

        site_id : str
            The three letter ICAO identifier of the station for which data should be
            downloaded.

        kwargs
            Arbitrary keyword arguments to use to initialize source

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        endpoint = cls()
        df = endpoint._get_data(time, site_id, **kwargs)
        return df

    def _get_data(self, time, site_id, region='naconf'):
        r"""Download and parse upper air observations from an online archive.

        Parameters
        ----------
        time : datetime
            The date and time of the desired observation.

        site_id : str
            The three letter ICAO identifier of the station for which data should be
            downloaded.

        region
            Region to request data from

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        raw_data, meta_data = self._get_data_raw(time, site_id, region)
        col_names = ['pressure', 'range', 'temperature', 'dewpoint', 'direction', 'speed']
        df = pd.read_fwf(raw_data, skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)
        df['u_wind'], df['v_wind'] = get_wind_components(df['speed'],
                                                         np.deg2rad(df['direction']))

        # Drop any rows with all NaN values for T, Td, winds
        df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed',
                               'u_wind', 'v_wind'), how='all').reset_index(drop=True)

        # Add unit dictionary
        df.units = {'pressure': 'hPa',
                    'range': 'meter',
                    'temperature': 'degC',
                    'dewpoint': 'degC',
                    'direction': 'degrees',
                    'speed': 'knot',
                    'u_wind': 'knot',
                    'v_wind': 'knot'}
        for item in list(meta_data.split('\n'))[1:-1]:
            var, value = item.split(': ')
            df._metadata.append({var.strip(): value})
        return df

    def _get_data_raw(self, time, site_id, region='naconf'):
        """Download data from the University of Wyoming's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded
        region : str
            The region in which the station resides. Defaults to `naconf`.

        Returns
        -------
        a file-like object from which to read the data

        """
        path = ('?region={region}&TYPE=TEXT%3ALIST'
                '&YEAR={time:%Y}&MONTH={time:%m}&FROM={time:%d%H}&TO={time:%d%H}'
                '&STNM={stid}').format(region=region, time=time, stid=site_id)

        resp = self.get_path(path)
        # See if the return is valid, but has no data
        if resp.text.find('Can\'t') != -1:
            raise ValueError(
                'No data available for {time:%Y-%m-%d %HZ} from region {region} '
                'for station {stid}.'.format(time=time, region=region,
                                             stid=site_id))

        soup = BeautifulSoup(resp.text, 'html.parser')
        return StringIO(soup.find_all('pre')[0].contents[0]), soup.find_all('pre')[1].contents[0]



def wyoming_pandas_to_dict(df):
    # extract metadata
    metadata = {k: v for d in df._metadata for k, v in d.items()}
    sounding_time = metadata['Observation time']
    date_sounding = datetime.datetime(int('20' + sounding_time[0:2]), int(sounding_time[2:4]), int(sounding_time[4:6]),

                                      int(sounding_time[7:9], int(sounding_time[9:11])))
    # build dictionary
    sounding = {}
    sounding['dimlabel'] = ['range']
    sounding['range'] = df['range'].values
    sounding['speed'] = (df['speed'].values * units.units('knots')).to_base_units().magnitude
    sounding['time'] = h.dt_to_ts(date_sounding)
    sounding['u_wind'] = (df['u_wind'].values * units.units('knots')).to_base_units().magnitude
    sounding['v_wind'] = (df['v_wind'].values * units.units('knots')).to_base_units().magnitude
    sounding['dewpoint'] = df['dewpoint']
    sounding['direction'] = df['direction']
    sounding['pressure'] = df['pressure']
    sounding['temperature'] = df['temperature']

    return sounding

def get_sounding(date, station_identifier):
    """Download Sounding from Uni Wyoming

        Args:
            date (datetime) of sounding of interest
            station_identifier (str), e.g. "SCCI" is Punta Arenas
        Returns:
            A dictionary containing the sounding data. More metadata (CAPE etc.) can be added later.

        """
    df = WyomingUpperAir.request_data(date, station_identifier)
    sounding = wyoming_pandas_to_dict(df)

    return sounding