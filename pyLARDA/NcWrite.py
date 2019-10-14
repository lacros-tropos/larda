import subprocess

import datetime
import netCDF4
import numpy as np

import pyLARDA.helpers as h


def generate_cloudnet_input_LIMRAD94(data, path, **kwargs):
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

    range_offsets = np.ones(no_chirps, dtype=np.uint32)
    for iC in range(no_chirps - 1):
        range_offsets[iC + 1] = range_offsets[iC] + data['C' + str(iC + 1) + 'Range']['var'][0].shape

    nc_add_variable(ds, val=range_offsets, dimension=('chirp',),
                    var_name='range_offsets', type=np.uint32,
                    long_name='chirp sequences start index array in altitude layer array', unit='[-]')

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
        var = nc_ds.createVariable(kwargs['var_name'], kwargs['type'], kwargs['dimension'], fill_value=-999.0)
        var[:] = kwargs['val']
        var.long_name = kwargs['long_name']
        var.unit = kwargs['unit']
    except Exception as e:
        raise e
