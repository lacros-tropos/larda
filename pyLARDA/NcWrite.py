import datetime
import netCDF4
import numpy as np
import pyLARDA.helpers as h
import time
import git


def export_spectra2nc(data, larda_git_path='', system='', path='', **kwargs):
    """
    This routine generates an hourly NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        system (string): name of the radar system
        path (string): path where the NetCDF file is stored
    """


    hour_bias = kwargs['hour_bias'] if 'hour_bias' in kwargs else 0


    dt_start = h.ts_to_dt(data['time'][0])
    dt_end = h.ts_to_dt(data['time'][-1])
    ds_name = path + f'{dt_start:%Y%m%d-%H%M}-{dt_end:%H%M}_{system}_spectra.nc'

    repo = git.Repo(larda_git_path)
    sha = repo.head.object.hexsha

    print('open file: ', ds_name)

    with netCDF4.Dataset(ds_name, "w", format="NETCDF4") as ds:

        ds.git_description = f'GIT commit ID  {sha}'
        ds.description = f'{system} calibrated Doppler spectra'
        ds.history = 'Created ' + time.ctime(time.time())
        ds.system = system
        ds.location = data['location']
        ds._FillValue = data['fill_value']

        ds.createDimension('chirp', data['chirps'])  # add variable number of chirps later
        ds.createDimension('time', data['time'].size)
        ds.createDimension('range', data['range'].size)
        ds.createDimension('velocity', data['velocity'].shape[1])

        nc_add_variable(
            ds,
            val=data['coordinates'][0],
            dimension=(),
            var_name='latitude',
            type=np.float32,
            long_name='GPS latitude',
            unit='deg'
        )
        nc_add_variable(
            ds,
            val=data['coordinates'][1],
            dimension=(),
            var_name='longitude',
            type=np.float32,
            long_name='GPS longitude',
            unit='deg'
        )
        nc_add_variable(
            ds,
            val=data['altitude'],
            dimension=(),
            var_name='altitude',
            type=np.float32,
            long_name='altitude, i.e. height above sea level',
            unit='m'
        )
        nc_add_variable(
            ds,
            val=data['time'],
            dimension=('time',),
            var_name='time',
            type=np.float64,
            long_name='Unix Time - seconds since 01.01.1970 00:00 UTC',
            unit='sec'
        )
        nc_add_variable(
            ds,
            val=data['rg_offsets'],
            dimension=('chirp',),
            var_name='rg_offsets',
            type=np.float32,
            long_name='Range Indices for next chirp sequence.',
            unit='-'
        )
        nc_add_variable(
            ds,
            val=data['range'],
            dimension=('range',),
            var_name='range',
            type=np.float32,
            long_name='range',
            unit='m'
        )
        nc_add_variable(
            ds,
            val=data['velocity'],
            dimension=('chirp', 'velocity',),
            var_name=f'velocity',
            type=np.float32,
            long_name='velocity vectors for each chirp',
            unit='m s-1'
        )
        nc_add_variable(
            ds,
            val=data['nyquist_velocity'],
            dimension=('chirp',),
            var_name='nyquist_velocity',
            type=np.float32,
            long_name='Unambiguous Doppler velocity (+/-)',
            unit='m s-1'
        )
        nc_add_variable(
            ds,
            val=data['doppler_spectrum'],
            dimension=('time', 'range', 'velocity'),
            var_name='doppler_spectrum',
            type=np.float32,
            long_name='Doppler spectrum, if dual polarization radar: doppler_spectrum = vertical + horizontal polarization',
            unit='mm6 m-3 (m s-1)-1'
        )
        try:
            nc_add_variable(
                ds,
                val=data['covariance_spectrum_re'],
                dimension=('time', 'range', 'velocity'),
                var_name='covariance_spectrum_re',
                type=np.float32,
                long_name='Real part of covariance spectrum',
                unit='mm6 m-3'
            )
        except KeyError:
            print('skip writing real part of covariance spectrum')

            try:
                nc_add_variable(
                ds,
                val=data['covariance_spectrum_im'],
                dimension=('time', 'range', 'velocity'),
                var_name='covariance_spectrum_im',
                type=np.float32,
                long_name='Imaginary part of covariance spectrum',
                unit='mm6 m-3'
            )
            except KeyError:
                print('skip writing imaginary part of covariance spectrum')

        try:
            nc_add_variable(
                ds,
                val=data['sensitivity_limit'],
                dimension=('time', 'range'),
                var_name='sensitivity_limit',
                type=np.float32,
                long_name='Sensitivity limit, if dual polarization radar: sensitivity_limit = vertical + horizontal polarization',
                unit='mm6 m-3'
            )
        except KeyError:
            print('skip writing sensitivity limit')
        try:
            nc_add_variable(
                ds,
                val=data['doppler_spectrum_h'],
                dimension=('time', 'range', 'velocity'),
                var_name='doppler_spectrum_h',
                type=np.float32,
                long_name='Doppler spectrum, horizontal polarization only',
                unit='mm6 m-3   '
            )
            nc_add_variable(
                ds,
                val=data['sensitivity_limit_h'],
                dimension=('time', 'range'),
                var_name='sensitivity_limit_h',
                type=np.float32,
                long_name='Sensitivity limit for horizontal polarization',
                unit='mm6 m-3 (m s-1)-1'
            )
        except KeyError:
            print('Skip writing horizontal polarization.')

    return 0


def export_spectra_to_nc(data, system='', path='', **kwargs):
    """
    This routine generates an hourly NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        system (string): name of the radar system
        path (string): path where the NetCDF file is stored
    """

    no_chirps = len(data['rg_offsets']) - 1

    dt_start = h.ts_to_dt(data['ts'][0])
    dt_end = h.ts_to_dt(data['ts'][-1])
    ds_name = path + '{}-{}_{}_spectra.nc'.format(dt_start.strftime("%Y%m%d-%H%M"), dt_end.strftime("%H%M"), system)

    print('open file: ', ds_name)

    with netCDF4.Dataset(ds_name, "w", format="NETCDF4") as ds:

        # ds.commit_id = subprocess.check_output(["git", "describe", "--always"]).rstrip()
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
                        var_name='latitude', type=np.float32, long_name='GPS latitude', unit='deg')
        nc_add_variable(ds, val=data['paraminfo']['coordinates'][1], dimension=(),
                        var_name='longitude', type=np.float32, long_name='GPS longitude', unit='deg')
        nc_add_variable(ds, val=data['ts'], dimension=('time',),
                        var_name='time', type=np.float64, long_name='Unix Time - seconds since 01.01.1970 00:00 UTC', unit='sec')

        nc_add_variable(ds, val=data['rg_offsets'][:no_chirps], dimension=('chirp',),
                        var_name='rg_offsets', type=np.float32, long_name='Range Indices when Chirp shifts ', unit='-')

        for ic in range(no_chirps):
            nc_add_variable(ds, val=data['rg'][ic], dimension=(f'C{ic + 1}range',),
                            var_name=f'C{ic + 1}range', type=np.float32, long_name='range', unit='m')
            nc_add_variable(ds, val=data['vel'][ic], dimension=(f'C{ic + 1}velocity',),
                            var_name=f'C{ic + 1}vel', type=np.float32, long_name='velocity', unit='m s-1')
            nc_add_variable(ds, val=data['var'][ic], dimension=('time', f'C{ic + 1}range', f'C{ic + 1}velocity'),
                            var_name=f'C{ic + 1}Zspec', type=np.float32,
                            long_name=f'Doppler spectrum at vertical+horizontal polarization: Chirp {ic + 1}', unit='mm6 m-3')
            nc_add_variable(ds, val=data['vel'][ic][-1], dimension=('chirp',),
                            var_name=f'C{ic + 1}DoppMax', type=np.float32, long_name='Unambiguous Doppler velocity (+/-)', unit='m s-1')

    return 0


def rpg_radar2nc(data, path, larda_git_path, **kwargs):
    """
    This routine generates a daily NetCDF4 file for the RPG 94 GHz FMCW radar 'LIMRAD94'.
    Args:
        data (dict): dictionary of larda containers
        path (string): path where the NetCDF file is stored
    """

    dt_start = h.ts_to_dt(data['Ze']['ts'][0])

    h.make_dir(path)
    site_name = kwargs['site'] if 'site' in kwargs else 'no-site'
    hour_bias = kwargs['hour_bias'] if 'hour_bias' in kwargs else 0
    cn_version = kwargs['version'] if 'version' in kwargs else 'pyhon'
    ds_name = f'{path}/{h.ts_to_dt(data["Ze"]["ts"][0]):%Y%m%d}-{site_name}-limrad94.nc'
    ncvers = '4'

    repo = git.Repo(larda_git_path)
    sha = repo.head.object.hexsha

    with netCDF4.Dataset(ds_name, 'w', format=f'NETCDF{ncvers}') as ds:
        ds.Convention = 'CF-1.0'
        ds.location = data['Ze']['paraminfo']['location']
        ds.system = data['Ze']['paraminfo']['system']
        ds.version = f'Variable names and dimensions prepared for Cloudnet {kwargs["version"]} version'
        ds.title = 'LIMRAD94 (SLDR) Doppler Cloud Radar, calibrated Input for Cloudnet'
        ds.institution = 'Leipzig Institute for Meteorology (LIM), Leipzig, Germany'
        ds.source = '94 GHz Cloud Radar LIMRAD94\nRadar type: Frequency Modulated Continuous Wave,\nTransmitter power 1.5 W typical (solid state ' \
                    'amplifier)\nAntenna Type: Bi-static Cassegrain with 500 mm aperture\nBeam width: 0.48deg FWHM'
        ds.reference = 'W Band Cloud Radar LIMRAD94\nDocumentation and User Manual provided by manufacturer RPG Radiometer Physics GmbH\n' \
                       'Information about system also available at https://www.radiometer-physics.de/'
        ds.calibrations = f'remove Precip. ghost: {kwargs["ghost_echo_1"]}\n, remove curtain ghost: {kwargs["ghost_echo_2"]}\n' \
                          f'despeckle: {kwargs["despeckle"]}\n, number of standard deviations above noise: {kwargs["NF"]}\n'

        ds.git_description = f'GIT commit ID  {sha}'
        ds.description = 'Concatenated data files of LIMRAD 94GHz - FMCW Radar, used as input for Cloudnet processing, ' \
                         'filters applied: ghos-echo, despeckle, use only main peak'
        ds.history = 'Created ' + time.ctime(time.time())
        ds._FillValue = data['Ze']['paraminfo']['fill_value']

        ds.day = dt_start.day
        ds.month = dt_start.month
        ds.year = dt_start.year

        # ds.commit_id = subprocess.check_output(["git", "describe", "--always"]) .rstrip()
        ds.history = 'Created ' + time.ctime(time.time()) + '\nfilters applied: ghost-echo, despeckle, main peak only'

        Ze_str = 'Zh' if cn_version == 'python' else 'Ze'
        vel_str = 'v' if cn_version == 'python' else 'vm'
        width_str = 'width' if cn_version == 'python' else 'sigma'
        dim_tupel = ('time', 'range') if cn_version == 'python' else ('range', 'time')

        n_chirps = len(data['no_av'])
        ds.createDimension('chirp', n_chirps)
        ds.createDimension('time', data['Ze']['ts'].size)
        ds.createDimension('range', data['Ze']['rg'].size)

        if cn_version == 'matlab':
            for ivar in ['Ze', 'VEL', 'sw', 'ldr', 'kurt', 'skew']:
                data[ivar]['var'] = data[ivar]['var'].T

        # coordinates
        nc_add_variable(
            ds,
            val=94.0,
            dimension=(),
            var_name='frequency',
            type=np.float32,
            long_name='Radar frequency',
            units='GHz'
        )

        nc_add_variable(
            ds,
            val=256,
            dimension=(),
            var_name='Numfft',
            type=np.float32,
            long_name='Number of points in FFT',
            units=''
        )

        nc_add_variable(
            ds,
            val=np.mean(data['MaxVel']['var']),
            dimension=(),
            var_name='NyquistVelocity',
            type=np.float32,
            long_name='Mean (over all chirps) Unambiguous Doppler velocity (+/-)',
            units='m s-1'
        )

        nc_add_variable(
            ds,
            val=data['Ze']['paraminfo']['altitude'],
            dimension=(),
            var_name='altitude',
            type=np.float32,
            long_name='Height of instrument above mean sea level',
            units='m'
        )

        nc_add_variable(
            ds,
            val=data['Ze']['paraminfo']['coordinates'][0],
            dimension=(),
            var_name='latitude',
            type=np.float32,
            long_name='latitude',
            units='degrees_north'
        )

        nc_add_variable(
            ds,
            val=data['Ze']['paraminfo']['coordinates'][1],
            dimension=(),
            var_name='longitude',
            type=np.float32,
            long_name='longitude',
            units='degrees_east'
        )

        if 'version' in kwargs and cn_version == 'python':
            nc_add_variable(
                ds,
                val=data['no_av'],
                dimension=('chirp',),
                var_name='NumSpectraAveraged',
                type=np.float32,
                long_name='Number of spectral averages',
                units=''
            )

        # time and range variable
        # convert to time since midnight
        if cn_version == 'python':
            ts = np.subtract(data['Ze']['ts'], datetime.datetime(dt_start.year, dt_start.month, dt_start.day, hour_bias, 0, 0).timestamp())
            ts_str = 'Decimal hours from midnight UTC to the middle of each day'
            ts_unit = f'hours since {dt_start:%Y-%m-%d} 00:00:00 +00:00 (UTC)'
            rg = data['Ze']['rg'] / 1000.0
        elif cn_version == 'matlab':
            ts = np.subtract(data['Ze']['ts'], datetime.datetime(2001, 1, 1, hour_bias, 0, 0).timestamp())
            ts_str = 'Seconds since 1st January 2001 00:00 UTC'
            ts_unit = 'sec'
            rg = data['Ze']['rg']
        else:
            raise ValueError('Wrong version selected! version to "matlab" or "python"!')

        nc_add_variable(ds, val=ts, dimension=('time',), var_name='time', type=np.float64, long_name=ts_str, units=ts_unit)
        nc_add_variable(ds, val=rg, dimension=('range',), var_name='range', type=np.float32,
                        long_name='Range from antenna to the centre of each range gate', units='km')
        nc_add_variable(ds, val=data['Azm']['var'], dimension=('time',), var_name='azimuth', type=np.float32,
                        long_name='Azimuth angle from north', units='degree')
        nc_add_variable(ds, val=data['Elv']['var'], dimension=('time',), var_name='elevation', type=np.float32,
                        long_name='elevation angle. 90 degree is vertical direction.', units='degree')

        # chirp dependent variables
        nc_add_variable(ds, val=data['MaxVel']['var'][0], dimension=('chirp',),
                        var_name='DoppMax', type=np.float32, long_name='Unambiguous Doppler velocity (+/-)', units='m s-1')

        # index plus (1 to n) for Matlab indexing
        nc_add_variable(ds, val=data['rg_offsets'], dimension=('chirp',),
                        var_name='range_offsets', type=np.uint32,
                        long_name='chirp sequences start index array in altitude layer array', units='-')

        # 1D variables
        nc_add_variable(ds, val=data['bt']['var'], dimension=('time',),
                        var_name='bt', type=np.float32, long_name='Direct detection brightness temperature', units='K')

        nc_add_variable(ds, val=data['LWP']['var'], dimension=('time',),
                        var_name='lwp', type=np.float32, long_name='Liquid water path', units='g m-2')

        nc_add_variable(ds, val=data['rr']['var'], dimension=('time',),
                        var_name='rain', type=np.float32, long_name='Rain rate from weather station', units='mm h-1')

        nc_add_variable(ds, val=data['SurfRelHum']['var'], dimension=('time',),
                        var_name='SurfRelHum', type=np.float32, long_name='Relative humidity from weather station', units='%')

        # 2D variables
        nc_add_variable(ds, val=data['Ze']['var'], dimension=dim_tupel, var_name=Ze_str, type=np.float32,
                        long_name='Radar reflectivity factor', units='mm6 m-3', plot_range=data['Ze']['var_lims'], plot_scale='linear',
                        comment='Calibrated reflectivity. Calibration convention: in the absence of attenuation, '
                                'a cloud at 273 K containing one million 100-micron droplets per cubic metre will '
                                'have a reflectivity of 0 dBZ at all frequencies.')

        nc_add_variable(ds, val=data['VEL']['var'], dimension=dim_tupel, plot_range=data['VEL']['var_lims'], plot_scale='linear',
                        var_name=vel_str, type=np.float32, long_name='Doppler velocity', units='m s-1', unit_html='m s<sup>-1</sup>',
                        comment='This parameter is the radial component of the velocity, with positive velocities are away from the radar.',
                        folding_velocity=data['MaxVel']['var'].max())

        nc_add_variable(ds, val=data['sw']['var'], dimension=dim_tupel, plot_range=data['sw']['var_lims'], lot_scale='logarithmic',
                        var_name=width_str, type=np.float32, long_name='Spectral width', units='m s-1', unit_html='m s<sup>-1</sup>',
                        comment='This parameter is the standard deviation of the reflectivity-weighted velocities in the radar pulse volume.')

        nc_add_variable(ds, val=data['ldr']['var'], dimension=dim_tupel, plot_range=[-30.0, 0.0],
                        var_name='ldr', type=np.float32, long_name='Linear depolarisation ratio', units='dB',
                        comment='This parameter is the ratio of cross-polar to co-polar reflectivity.')

        nc_add_variable(ds, val=data['kurt']['var'], dimension=dim_tupel, plot_range=data['kurt']['var_lims'],
                        var_name='kurt', type=np.float32, long_name='Kurtosis', units='linear')

        nc_add_variable(ds, val=data['skew']['var'], dimension=dim_tupel, plot_range=data['skew']['var_lims'],
                        var_name='Skew', type=np.float32, long_name='Skewness', units='linear')

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

    with netCDF4.Dataset(ds_name, "w", format="NETCDF4") as ds:
        # ds.commit_id = subprocess.check_output(["git", "describe", "--always"]) .rstrip()
        ds.description = 'Concatenated data files of LIMRAD 94GHz - FMCW Radar, used as input for Cloudnet processing, ' \
                         'filters applied: ghos-echo, despeckle, use only main peak'
        ds.history = 'Created ' + time.ctime(time.time())
        ds.source = data['Ze']['paraminfo']['location']
        ds._FillValue = data['Ze']['paraminfo']['fill_value']

        ds.createDimension('chirp', no_chirps)  # add variable number of chirps later
        ds.createDimension('time', data['Ze']['ts'].size)
        ds.createDimension('range', data['Ze']['rg'].size)

        # coordinates
        nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][0], dimension=(),
                        var_name='latitude', type=np.float32, long_name='GPS latitude', unit='deg')

        nc_add_variable(ds, val=data['Ze']['paraminfo']['coordinates'][1], dimension=(),
                        var_name='longitude', type=np.float32, long_name='GPS longitude', unit='deg')

        # time and range variable
        # convert to time since 20010101
        ts = np.subtract(data['Ze']['ts'], datetime.datetime(2001, 1, 1, 0, 0, 0).timestamp())
        nc_add_variable(ds, val=ts, dimension=('time',),
                        var_name='time', type=np.float64, long_name='Seconds since 01.01.2001 00:00 UTC', unit='sec')

        nc_add_variable(ds, val=data['Ze']['rg'], dimension=('range',),
                        var_name='range', type=np.float32, long_name='range', unit='m')

        # 2D variables
        nc_add_variable(ds, val=data['Ze']['var'].T, dimension=('range', 'time',),
                        var_name='Ze', type=np.float32, long_name='Equivalent radar reflectivity factor', unit='mm^6/m^3')

        nc_add_variable(ds, val=data['VEL']['var'].T, dimension=('range', 'time',),
                        var_name='vm', type=np.float32, long_name='Mean Doppler velocity', unit='m/s')

        nc_add_variable(ds, val=data['sw']['var'].T, dimension=('range', 'time',),
                        var_name='sigma', type=np.float32, long_name='Spectrum width', unit='m/s')

        nc_add_variable(ds, val=data['ldr']['var'].T, dimension=('range', 'time',),
                        var_name='ldr', type=np.float32, long_name='Slanted linear depolarization ratio', unit='dB')

        nc_add_variable(ds, val=data['kurt']['var'].T, dimension=('range', 'time',),
                        var_name='kurt', type=np.float32, long_name='Kurtosis', unit='[linear]')

        nc_add_variable(ds, val=data['skew']['var'].T, dimension=('range', 'time',),
                        var_name='Skew', type=np.float32, long_name='Skewness', unit='[linear]')

        nc_add_variable(ds, val=data['DiffAtt']['var'].T, dimension=('range', 'time',),
                        var_name='DiffAtt', type=np.float32, long_name='Differential attenuation', unit='dB/km')

        # 1D variables
        nc_add_variable(ds, val=data['bt']['var'], dimension=('time',),
                        var_name='bt', type=np.float32, long_name='Direct detection brightness temperature', unit='K')

        nc_add_variable(ds, val=data['LWP']['var'], dimension=('time',),
                        var_name='lwp', type=np.float32, long_name='Liquid water path', unit='g/m^2')

        nc_add_variable(ds, val=data['rr']['var'], dimension=('time',),
                        var_name='rain', type=np.float32, long_name='Rain rate from weather station', unit='mm/h')

        nc_add_variable(ds, val=data['SurfRelHum']['var'], dimension=('time',),
                        var_name='SurfRelHum', type=np.float32, long_name='Relative humidity from weather station', unit='%')

        # chirp dependent variables
        nc_add_variable(ds, val=data['MaxVel']['var'][0], dimension=('chirp',),
                        var_name='DoppMax', type=np.float32, long_name='Unambiguous Doppler velocity (+/-)', unit='m/s')

        range_offsets = np.ones(no_chirps, dtype=np.uint)
        for iC in range(no_chirps - 1):
            range_offsets[iC + 1] = range_offsets[iC] + data['C' + str(iC + 1) + 'Range']['var'][0].shape

        nc_add_variable(ds, val=range_offsets, dimension=('chirp',),
                        var_name='range_offsets', type=np.uint,
                        long_name='chirp sequences start index array in altitude layer array', unit='[-]')

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
        _fillvalue = -999.0 if kwargs['type'] == np.float32 else 4294966297
        var = nc_ds.createVariable(kwargs['var_name'], kwargs['type'], kwargs['dimension'], fill_value=_fillvalue)
        var[:] = kwargs['val']

        key_list = ['long_name', 'units', 'plot_range', 'folding_velocity', 'plot_scale']

        #        if len(kwargs['dimension']) > 0:
        #            kwargs['_FillValue'] = -999.0
        #            key_list.append('_FillValue')

        var.setncatts({f'{key}': kwargs[key] for key in key_list if key in kwargs})


    except Exception as e:
        raise e

    return var
