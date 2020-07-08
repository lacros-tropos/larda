import subprocess
import datetime
import time
import netCDF4
import numpy as np
import os

import pyLARDA.helpers as h


def export_spectra_to_nc(data, system='', path='', **kwargs):
    """
    This routine generates an hourly NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        system (string): name of the radar system
        path (string): path where the NetCDF file is stored
    """

    no_chirps = len(data['rg_offsets'])-1

    dt_start = h.ts_to_dt(data['ts'][0])
    dt_end = h.ts_to_dt(data['ts'][-1])
    ds_name = path + '{}-{}_{}_spectra.nc'.format(dt_start.strftime("%Y%m%d-%H%M"), dt_end.strftime("%H%M"), system)

    print('open file: ', ds_name)

    with netCDF4.Dataset(ds_name, "w", format="NETCDF4") as ds:

        #ds.commit_id = subprocess.check_output(["git", "describe", "--always"]).rstrip()
        ds.description = '{} calibrated Doppler spectra'.format(system)
        ds.history = 'Created ' + time.ctime(time.time())
        ds.system = system
        ds.location = data['paraminfo']['location']
        ds._FillValue = data['paraminfo']['fill_value']

        ds.createDimension('chirp', no_chirps)  # add variable number of chirps later
        ds.createDimension('time', data['ts'].size)
        for ic in range(no_chirps):
            ds.createDimension(f'C{ic + 1}range', data['rg'][ic].size)
            ds.createDimension(f'C{ic + 1}velocity', data['vel'][ic].size)

        nc_add_variable(ds, val=data['paraminfo']['coordinates'][0], dimension=(),
                        var_name='latitude', type=np.float32, long_name='GPS latitude', units='deg')
        nc_add_variable(ds, val=data['paraminfo']['coordinates'][1], dimension=(),
                        var_name='longitude', type=np.float32, long_name='GPS longitude', units='deg')
        nc_add_variable(ds, val=data['ts'], dimension=('time',),
                        var_name='time', type=np.float64, long_name='Unix Time - seconds since 01.01.1970 00:00 UTC', units='sec')

        nc_add_variable(ds, val=data['rg_offsets'][:no_chirps], dimension=('chirp',),
                        var_name='rg_offsets', type=np.float32, long_name='Range Indices when Chirp shifts ', units='-')

        for ic in range(no_chirps):
            nc_add_variable(ds, val=data['rg'][ic], dimension=(f'C{ic + 1}range',),
                            var_name=f'C{ic + 1}range', type=np.float32, long_name='range', units='m')
            nc_add_variable(ds, val=data['vel'][ic], dimension=(f'C{ic + 1}velocity',),
                            var_name=f'C{ic + 1}vel', type=np.float32, long_name='velocity', units='m s-1')
            nc_add_variable(ds, val=data['var'][ic], dimension=('time', f'C{ic + 1}range', f'C{ic + 1}velocity'),
                            var_name=f'C{ic + 1}Zspec', type=np.float32,
                            long_name=f'Doppler spectrum at vertical+horizontal polarization: Chirp {ic + 1}', units='mm6 m-3')
            nc_add_variable(ds, val=data['vel'][ic][-1], dimension=('chirp',),
                            var_name=f'C{ic + 1}DoppMax', type=np.float32, long_name='Unambiguous Doppler velocity (+/-)', units='m s-1')


    return 0


def rpg_radar2nc(data, path, **kwargs):
    """
    This routine generates a daily NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        path (string): path where the NetCDF file is stored
    """
    import time

    dt_start = h.ts_to_dt(data['Ze']['ts'][0])

    h.make_dir(path)
    site_name = kwargs['site'] if 'site' in kwargs else 'no-site'
    hour_bias = kwargs['hour_bias'] if 'hour_bias' in kwargs else 0
    ds_name = path + '{}-{}-limrad94.nc'.format(h.ts_to_dt(data['Ze']['ts'][0]).strftime("%Y%m%d"), site_name)

    with netCDF4.Dataset(ds_name, 'w', format='NETCDF4') as ds:
        ds.Convention = 'CF-1.0'
        ds.location = data['Ze']['paraminfo']['location']
        ds.system = data['Ze']['paraminfo']['system']
        ds.title = 'LIMRAD94 (SLDR) Doppler Cloud Radar, calibrated Input for Cloudnetpy'
        ds.institution = 'Leipzig Institute for Meteorology (LIM), Leipzig, Germany'
        ds.source = '94 GHz Cloud Radar LIMRAD94\nRadar type: Frequency Modulated Continuous Wave,\nTransmitter power 1.5 W typical (solid state ' \
                    'amplifier)\nAntenna Type: Bi-static Cassegrain with 500 mm aperture\nBeam width: 0.48deg FWHM'
        ds.reference = 'W Band Cloud Radar LIMRAD94\nDocumentation and User Manual provided by manufacturer RPG Radiometer Physics GmbH\n' \
                       'Information about system also available at https://www.radiometer-physics.de/'
        ds.calibrations = f'remove Precip. ghost: {kwargs["rm_precip_ghost"]}, remove curtain ghost: {kwargs["filter_ghost_C1"]}\n' \
                          f'despeckle3d: {kwargs["do_despeckle3d"]}, despeckle2d: {kwargs["despeckle"]}\n' \
                          f'noise estimation: {kwargs["estimate_noise"]}, std above noise: {kwargs["NF"]}\n' \
                          f'main peak only: {kwargs["main_peak"]}'

        ds.day = dt_start.day
        ds.month = dt_start.month
        ds.year = dt_start.year

        # ds.commit_id = subprocess.check_output(["git", "describe", "--always"]) .rstrip()
        ds.history = 'Created ' + time.ctime(time.time()) + '\nfilters applied: ghost-echo, despeckle, main peak only'

        ds.createDimension('chirp', len(data['no_av']))
        ds.createDimension('time', data['Ze']['ts'].size)
        ds.createDimension('range', data['Ze']['rg'].size)

        # coordinates
        nc_add_variable(ds, val=94.0, dimension=(), var_name='frequency', type=np.float32, long_name='Radar frequency', units='GHz')
        nc_add_variable(ds, val=256, dimension=(), var_name='Numfft', type=np.float32, long_name='Number of points in FFT', units=' ')
        nc_add_variable(ds, val=np.mean(data['MaxVel']['var']), dimension=(), var_name='NyquistVelocity', type=np.float32,
                        long_name='Mean (over all chirps) Unambiguous Doppler velocity (+/-)', units='m s-1')

        nc_add_variable(ds, val=data['Ze']['paraminfo']['altitude'], dimension=(),
                        var_name='altitude', type=np.float32, long_name='Height of instrument above mean sea level', units='m')
        nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][0], dimension=(),
                        var_name='latitude', type=np.float32, long_name='latitude', units='degrees_north')
        nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][1], dimension=(),
                        var_name='longitude', type=np.float32, long_name='longitude', units='degrees_east')

        if 'version' in kwargs and kwargs['version'] == 'v3':
            nc_add_variable(ds, val=data['no_av'], dimension=('chirp',), var_name='NumSpectraAveraged', type=np.float32,
                            long_name='Number of spectral averages', units=' ')
        else:
            nc_add_variable(ds, val=data['no_av'], dimension=(), var_name='NumSpectraAveraged', type=np.float32,
                            long_name='Number of spectral averages', units=' ')
        # time and range variable
        # convert to time since midnight
        ts = np.subtract(data['Ze']['ts'], datetime.datetime(dt_start.year, dt_start.month, dt_start.day, hour_bias, 0, 0).timestamp())
        nc_add_variable(ds, val=ts, dimension=('time',), var_name='time', type=np.float64,
                        long_name='Decimal hours from midnight UTC to the middle of each day',
                        units=f'hours since {dt_start:%Y-%m-%d} 00:00:00 +00:00 (UTC)', axis='T')

        nc_add_variable(ds, val=data['Ze']['rg'] / 1000.0, dimension=('range',), var_name='range', type=np.float32,
                        long_name='Range from antenna to the centre of each range gate', units='km', axis='Z')
        nc_add_variable(ds, val=data['Azm']['var'], dimension=('time',), var_name='azimuth', type=np.float32,
                        long_name='Azimuth angle from north', units='degree', axis='Z')
        nc_add_variable(ds, val=data['Elv']['var'], dimension=('time',), var_name='elevation', type=np.float32,
                        long_name='elevation angle. 90 degree is vertical direction.', units='degree', axis='Z')

        # 2D variables
        nc_add_variable(ds, val=h.lin2z(data['Ze']['var']), dimension=('time', 'range'), var_name='Zh', type=np.float32,
                        long_name='Radar reflectivity factor', units='dBZ', plot_range=data['Ze']['var_lims'], plot_scale='linear',
                        comment='Calibrated reflectivity. Calibration convention: in the absence of attenuation, '
                                'a cloud at 273 K containing one million 100-micron droplets per cubic metre will '
                                'have a reflectivity of 0 dBZ at all frequencies.')

        nc_add_variable(ds, val=data['VEL']['var'], dimension=('time', 'range'), plot_range=data['VEL']['var_lims'], plot_scale='linear',
                        var_name='v', type=np.float32, long_name='Doppler velocity', units='m s-1', units_html='m s<sup>-1</sup>',
                        comment='This parameter is the radial component of the velocity, with positive velocities are away from the radar.',
                        folding_velocity=data['MaxVel']['var'].max())

        nc_add_variable(ds, val=data['sw']['var'], dimension=('time', 'range'), plot_range=data['sw']['var_lims'], plot_scale='logarithmic',
                        var_name='width', type=np.float32, long_name='Spectral width', units='m s-1', units_html='m s<sup>-1</sup>',
                        comment='This parameter is the standard deviation of the reflectivity-weighted velocities in the radar pulse volume.')

        nc_add_variable(ds, val=data['ldr']['var'], dimension=('time', 'range'), plot_range=[-30.0, 0.0], plot_scale='linear',
                        var_name='ldr', type=np.float32, long_name='Linear depolarisation ratio', units='dB',
                        comment='This parameter is the ratio of cross-polar to co-polar reflectivity.')

        nc_add_variable(ds, val=data['kurt']['var'], dimension=('time', 'range'), plot_range=data['kurt']['var_lims'], plot_scale='linear',
                        var_name='kurt', type=np.float32, long_name='Kurtosis', units='[linear]')

        nc_add_variable(ds, val=data['skew']['var'], dimension=('time', 'range'), plot_range=data['skew']['var_lims'], plot_scale='linear',
                        var_name='Skew', type=np.float32, long_name='Skewness', units='[linear]')

    print('save calibrated to :: ', ds_name)

    return 0


def rpg_radar2nc_old(data, path, **kwargs):
    """
    This routine generates a daily NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        path (string): path where the NetCDF file is stored
    """
    import time

    no_chirps = 3

    if 'time_frame' in kwargs:
        ds_name = path + kwargs['time_frame'] + '_LIMRAD94.nc'.format(h.ts_to_dt(data['Ze']['ts'][0]).strftime("%Y%m%d"),
                                                                      kwargs['time_frame'])
    else:
        ds_name = path + '{}_000000-240000_LIMRAD94.nc'.format(h.ts_to_dt(data['Ze']['ts'][0]).strftime("%Y%m%d"))

    ds = netCDF4.Dataset(ds_name, "w", format="NETCDF4")

    # ds.commit_id = subprocess.check_output(["git", "describe", "--always"]) .rstrip()
    ds.description = 'Concatenated data files of LIMRAD 94GHz - FMCW Radar, used as input for Cloudnet processing, ' \
                     'filters applied: ghos-echo, despeckle, use only main peak'
    ds.history = 'Created ' + time.ctime(time.time())
    ds.source = data['Ze']['paraminfo']['location']
    ds.FillValue = data['Ze']['paraminfo']['fill_value']

    ds.createDimension('chirp', no_chirps)  # add variable number of chirps later
    ds.createDimension('time', data['Ze']['ts'].size)
    ds.createDimension('range', data['Ze']['rg'].size)

    # coordinates
    nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][0], dimension=(),
                    var_name='latitude', type=np.float32, long_name='GPS latitude', units='deg')

    nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][1], dimension=(),
                    var_name='longitude', type=np.float32, long_name='GPS longitude', units='deg')

    # time and range variable
    # convert to time since 20010101
    ts = np.subtract(data['Ze']['ts'], datetime.datetime(2001, 1, 1, 0, 0, 0).timestamp())
    nc_add_variable(ds, val=ts, dimension=('time',),
                    var_name='time', type=np.float64, long_name='Seconds since 01.01.2001 00:00 UTC', units='sec')

    nc_add_variable(ds, val=data['Ze']['rg'], dimension=('range',),
                    var_name='range', type=np.float32, long_name='range', units='m')

    # 2D variables
    nc_add_variable(ds, val=data['Ze']['var'].T, dimension=('range', 'time',),
                    var_name='Ze', type=np.float32, long_name='Equivalent radar reflectivity factor', units='mm^6/m^3')

    nc_add_variable(ds, val=data['VEL']['var'].T, dimension=('range', 'time',),
                    var_name='vm', type=np.float32, long_name='Mean Doppler velocity', units='m/s')

    nc_add_variable(ds, val=data['sw']['var'].T, dimension=('range', 'time',),
                    var_name='sigma', type=np.float32, long_name='Spectrum width', units='m/s')

    nc_add_variable(ds, val=data['ldr']['var'].T, dimension=('range', 'time',),
                    var_name='ldr', type=np.float32, long_name='Slanted linear depolarization ratio', units='dB')

    nc_add_variable(ds, val=data['kurt']['var'].T, dimension=('range', 'time',),
                    var_name='kurt', type=np.float32, long_name='Kurtosis', units='[linear]')

    nc_add_variable(ds, val=data['skew']['var'].T, dimension=('range', 'time',),
                    var_name='Skew', type=np.float32, long_name='Skewness', units='[linear]')

    nc_add_variable(ds, val=data['DiffAtt']['var'].T, dimension=('range', 'time',),
                    var_name='DiffAtt', type=np.float32, long_name='Differential attenuation', units='dB/km')

    # 1D variables
    nc_add_variable(ds, val=data['bt']['var'], dimension=('time',),
                    var_name='bt', type=np.float32, long_name='Direct detection brightness temperature', units='K')

    nc_add_variable(ds, val=data['LWP']['var'], dimension=('time',),
                    var_name='lwp', type=np.float32, long_name='Liquid water path', units='g/m^2')

    nc_add_variable(ds, val=data['rr']['var'], dimension=('time',),
                    var_name='rain', type=np.float32, long_name='Rain rate from weather station', units='mm/h')

    nc_add_variable(ds, val=data['SurfRelHum']['var'], dimension=('time',),
                    var_name='SurfRelHum', type=np.float32, long_name='Relative humidity from weather station', units='%')

    # chirp dependent variables
    nc_add_variable(ds, val=data['MaxVel']['var'][0], dimension=('chirp',),
                    var_name='DoppMax', type=np.float32, long_name='Unambiguous Doppler velocity (+/-)', units='m/s')

    range_offsets = np.ones(no_chirps, dtype=np.uint32)
    for iC in range(no_chirps - 1):
        range_offsets[iC + 1] = range_offsets[iC] + data['C' + str(iC + 1) + 'Range']['var'][0].shape

    nc_add_variable(ds, val=range_offsets, dimension=('chirp',),
                    var_name='range_offsets', type=np.uint32,
                    long_name='chirp sequences start index array in altitude layer array', units='[-]')

    ds.close()

    print('save calibrated to :: ', ds_name)

    return 0


def nc_add_variable(nc_ds, **kwargs):
    """
    Helper function for adding a variable to a NetCDF file
    Args:
        nc_ds (NetCDF4 object): NetCDF data container with writing permission
        **var_name (string): variable name
        **type (numpy.uint32, numpy.float32): variable type
        **dimension(tuple): dimensionality of the variable
        **val (numpy.array): values of the variable
        **long_name (string): more detailed description of the variable
        **unit (string): variable unit
    """
    try:
        var = nc_ds.createVariable(kwargs['var_name'], kwargs['type'], kwargs['dimension'])
        var[:] = kwargs['val']
        if 'long_name' in kwargs: var.long_name = kwargs['long_name']
        if 'units' in kwargs: var.units = kwargs['units']
        if 'plot_range' in kwargs: var.plot_range = kwargs['plot_range']
        if 'folding_velocity' in kwargs: var.folding_velocity = kwargs['folding_velocity']
        if 'plot_scale' in kwargs: var.plot_scale = kwargs['plot_scale']
        if 'missing_value' in kwargs: var.missing_value = kwargs['missing_value']
    except Exception as e:
        raise e

    return var
