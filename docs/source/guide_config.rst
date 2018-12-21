
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
        systems = []
        cloudnet_stationname = 'punta-arenas'
        param_config_file = 'params_dacapo.toml'
        connectordump = '/home/larda/larda-connectordump/'
                                

Parameter config
----------------

The second level of configuration defines the parameters for each system in a file such as 
``params_dacapo.toml``.


.. note::

    implicitly it is assumed, that the timestamp in the filename
    is the beginning of measurements
