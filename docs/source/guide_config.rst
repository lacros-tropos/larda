
######################
Guide to config-files
######################

General remarks
---------------
Instead of ``.csv`` files als in previos versions, now ``.toml`` markup is used.
An overview is provided here https://github.com/toml-lang/toml


Campaigns config
----------------

A ``campaigns.toml`` config file is used to provide the general context of the data.

.. code-block:: none

    [lacros_dacapo]
        location = "Punta Arenas"
        coordinates = [-53.1354, -70.8845]
        altitude = 9
        duration = [["20181127", "today"]]
        systems = ["MIRA", "CLOUDNET"]
        cloudnet_stationname = 'punta-arenas'
        param_config_file = 'params_dacapo.toml'
        connectordump = '/home/larda/larda-connectordump/'
                                

Parameter config
----------------

The second level of configuration defines the parameters for each system in a file such as 
``params_dacapo.toml``. The file lists different systems, such as ``MIRA`` or ``CLOUDNET``.
Each systems' config has three parts.

path
^^^^

Defines the paths, where to find the (netcdf) files containing the parameters.

base_dir
    directory to start filesearch

matching_subdirs
    regex, that describes the subpaths (including filename) that are matching_subdirs

date_in_filename
    named groups in regex that identify the part of the filename, that contains the date

.. note::

    implicitly it is assumed, that the timestamp in the filename
    is the beginning of measurements

generic
^^^^^^^

Set of properties, that is used for each parameter as default if not specified explicitly for the parameter itself.

params
^^^^^^

Define the settings for each variable, that should be handled by larda.

time_variable
    name of the time variable in the netcdf file

time_conversion
    function that converts the time variable to unix timestamp

time_microsec_variable
    if given specifies the variable containing the microseconds

time_millisec_variable
    if given specifies the variable containing the milliseconds

range_variable
    name of the range variable in the netcdf file

range_conversion
    function that converts the range variable to meters. ``'none'`` if already given in that unit

var_conversion
    function that converts the variable variable to meters. ``'none'`` if no conversion is desired

colormap
    colormap to use by default

which_path
    name of the path definition that matches the files for this parameter

ncreader
    which reader to use

identifier_rg_unit
    name of the range unit attribute in the netcdf varibale

identifier_var_unit
    name of the var unit attribute in the netcdf varibale

identifier_var_lims
    name of the var limits attribute in the netcdf varibale

var_lims
    define limits of variable directly

var_name
    name of the variable

vel_variable
    velocity variable for reading spectra

dimorder
    toggle the order of dimensions (i.e. mira nc file)

identifier_history
    attribut in the netcdf file that is used to store the processing history




.. code-block:: none

    [add the example]
