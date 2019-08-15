
####################################
Data container and transformations
####################################


Data container format
---------------------
The connector provides the data loaded from the netcdf file as a data contianer (plain python ``dict`` with a specific set of keys).
As a rule of thumb the :meth:`pyLARDA.NcReader` returns the variable in linear units, the time as unix timestamp and the range in m.
Following keys are currently specified:

======================  =========================================================================
  Key                    Example                            
======================  =========================================================================
 ``ts``                  timestamps
 ``rg``                  ranges (optional)
 ``vel``                 velocity (of Doppler spectrum, optional)
 ``dimlabel``            eg ``['time']``, ``['time', 'range']``, ``['time', 'range', 'vel']``
                                              
 ``var``                 the actual data array
 ``mask``                a mask for the data
                                              
 ``paraminfo``           the info dict derived for the parameter config file (do not mutate)
 ``filename``            the source file
 ``rg_unit``             unit for the range
 ``var_unit``            unit for the variable
 ``var_lims``            limits for the plot
 ``colormap``            colormap to use
 ``plot_varconverter``
 ``file_history``        (opt) list of the processing histories of the original files
======================  =========================================================================



Transformations
---------------
Transformations operate on a single or several data container(s). 

.. code-block:: python

    # get a data_container
    MIRA_Zg = larda.read("MIRA","Zg", [dt_begin, dt_end], [0, 4000])
    # plot it
    fig, ax = pyLARDA.Transformations.plot_timeheight(
        MIRA_Zg, range_interval=[500, 3000], z_converter='lin2z')
    fig.savefig('MIRA_Z.png', dpi=250)
                                                                 
    # or for interpolation
    interpolated_container = pyLARDA.Transformations.interpolate2d(MIRA_Zg, new_time=array)
    interpolated_container = pyLARDA.Transformations.interpolate2d(MIRA_Zg, new_range=array)

    h.pprint(MIRA_Zg)


It is also possibledo do calculations on data containers or combine them without
loosing the meta-information:

.. code-block:: python

    def correct_Z_bias(data):
        var = data['var'] + h.z2lin(8.)
        return var, data['mask']

    Z_corrected = pyLARDA.Transformations.combine(correct_Z_bias, Z, {})


doc-strings
---------------

.. automodule:: pyLARDA.Transformations
   :members:
