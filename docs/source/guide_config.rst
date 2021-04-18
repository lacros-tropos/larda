
######################
Guide to config-files
######################

General remarks
---------------
Instead of ``.csv`` files als in previos versions, now ``.toml`` markup is used.
An overview is provided here https://github.com/toml-lang/toml


Setup sample files
-------------------
To illustrate the configuration, some sample files are needed.

.. code-block:: none

    # make a folder in the the directory as larda, larda-cfg, and larda-connectordump
    mkdir -p example-data/categorize/2018 && cd example-data/categorize/2018/
    wget http://devcloudnet.fmi.fi/cnet/limassol/processed/categorize/2018/20180208_limassol_categorize.nc
    cd ../../../
    mkdir -p example-data/classification/2018 && example-data/classification/2018
    wget http://devcloudnet.fmi.fi/cnet/limassol/products/classification/2018/20180208_limassol_classification.nc


Campaigns config
----------------

The  ``larda-cfg/campaigns.toml`` config file is used to provide the general context of the data.

.. code-block:: none

    [lacros_cycare_example]
        location = "Limassol"
        coordinates = [34.677, 33.038]
        altitude = 11
        mira_azi_zero = 154
        duration = [["20161018", "today"]]
        systems = ["CLOUDNET", "MIRA"]
        system_only.CLOUDNET =  [['20161101', '20190107'], ['20180101', '20180214']]
        system_only.MIRA =  [['20161101', '20180401']]
        cloudnet_stationname = 'limassol'
        info_text_loc = 'default'
        #info_text_loc = 'info_lacros.toml'       
        param_config_file = 'params_cycare_example.toml'
        connectordump = '/home/larda3/larda-connectordump/'




Parameter config
----------------

The second level of configuration defines the parameters for each system in a file such as 
``params_cycare_example.toml``. The file lists different systems, such as ``MIRA``, ``CLOUDNET`` or ``POLLYNET``.
Each systems' config has three parts. An example is given below.


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

meta.*
    dictionary of meta information extracted from variables, var attributes or global attributes


Example
^^^^^^^

The section for the Cloudnet configration in the ``params_cycare_example.toml`` might look like below.
The absolute paths in ``base_dir`` will likely have to be adapted.

.. code-block:: none

    [CLOUDNET]
        [CLOUDNET.path.categorize]
            # mastering regex (here to exclude ppi and stuff)
            base_dir = '/home/larda3/example-data/categorize/'
            matching_subdirs = '(\d{4}\/\d{8}.*.nc)'
            date_in_filename = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
        [CLOUDNET.path.productsclass]
            # mastering regex (here to exclude ppi and stuff)
            base_dir = '/home/larda3/example-data/classification/'
            matching_subdirs = '(\d{4}\/\d{8}.*.nc)'
            date_in_filename = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
        [CLOUDNET.generic]
            # this general settings need to be handed down to the params
            time_variable = 'time'
            range_variable = 'height'
            colormap = "gist_rainbow"
            time_conversion = 'beginofday'
            range_conversion = 'sealevel2range'
            var_conversion = 'none'
            ncreader = 'timeheight'
            # if identifier is given read from ncfile, else define here
            identifier_rg_unit = 'units'
            identifier_var_unit = 'units'
            identifier_var_lims = 'plot_range'
            identifier_fill_value = 'missing_value'
            #var_lims = [-40, 20]
            meta.version = "gattr.software_version"
            meta.history = "gattr.history"
            meta.source = "vattr.source"
            meta.latitude = "var.latitude"
        [CLOUDNET.params.Z]
            variable_name = 'Z'
            which_path = 'categorize'
            var_conversion = 'z2lin'
        [CLOUDNET.params.LDR]
            variable_name = 'ldr'
            which_path = 'categorize'
            var_conversion = 'z2lin'
        [CLOUDNET.params.T]
            variable_name = 'temperature'
            which_path = 'categorize'
            range_variable = 'model_height'
        [CLOUDNET.params.beta]
            variable_name = 'beta'
            which_path = 'categorize'
        [CLOUDNET.params.depol]
            variable_name = 'lidar_depolarisation'
            which_path = 'categorize'    
            var_unit = '%'
            var_lims = [0.0, 0.3]
        [CLOUDNET.params.CLASS]
            variable_name = 'target_classification'
            which_path = 'productsclass'
            var_unit = ""
            var_lims = [0, 10]
            colormap = 'cloudnet_target'
            fill_value = -99


.. note::

    ``var_conversion`` allows for chained functions, such as ``var_conversion = 'z2lin,extrfromaxis2(0)'``.
    See :meth:`pyLARDA.helpers.get_converter_array`.


A template option is available for repeating datasets in different campaigns:

.. code-block:: none

    [CLOUDNET]
        template = 'temp_cloudnet.toml'
        [CLOUDNET.path.categorize]
            # mastering regex (here to exclude ppi and stuff)
            base_dir = '/home/larda3/example-data/categorize/'
            matching_subdirs = '(\d{4}\/\d{8}.*.nc)'
            date_in_filename = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'
        [CLOUDNET.path.productsclass]
            # mastering regex (here to exclude ppi and stuff)
            base_dir = '/home/larda3/example-data/classification/'
            matching_subdirs = '(\d{4}\/\d{8}.*.nc)'
            date_in_filename = '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_'


The ``generic`` and ``params`` section are then defined in :file:`larda-cfg/temp_cloudnet.toml`.
In the :class:`pyLARDA.ParameterInfo.ParameterInfo`, the template is updated with the campaign configuration.
Hence, single ``generic`` or ``params`` configurations in the template can be overwritten. 

The configuration can be checked by running ``python3 ListCollector.py``
Afterwards the connectordump at ``larda-connectordump/lacros_cycare_example/connector_CLOUDNET.json``
should look similar to

.. code-block:: none

    {
    "categorize": [
        [
        [
            "20180208-000000",
            "20180209-000000"
        ],
        "./2018/20180208_limassol_categorize.nc"
        ]
    ],
    "productsclass": [
        [
        [
            "20180208-000000",
            "20180209-000000"
        ],
        "./2018/20180208_limassol_classification.nc"
        ]
    ]
    }