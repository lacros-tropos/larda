
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
 ``ts``                  ``numpy.array`` containing unix timestamps
 ``rg``                  ``numpy.array`` containing ranges (optional)
 ``vel``                 velocity (of Doppler spectrum, optional)
 ``dimlabel``            eg ``['time']``, ``['time', 'range']``, ``['time', 'range', 'vel']``
                                              
 ``var``                 the actual data as ``numpy.array``
 ``mask``                a bool mask ``numpy.array`` for the data
                                              
 ``paraminfo``           the info dict derived for the parameter config file (do not mutate)
 ``filename``            the source file
 ``rg_unit``             unit for the range
 ``var_unit``            unit for the variable
 ``var_lims``            limits for the plot
 ``colormap``            colormap to use
 ``plot_varconverter``
 ``meta``                dict of meta data read with the meta.* tags
 ``var_definition``      (opt) dict definition of the variable (e.g. cloudnet class flags)
======================  =========================================================================


Transformations
---------------
Transformations operate on a single or several data container(s). 

.. hint::

    As a rule of thumb, transformations either return a data container or a plot.



Interpolation
^^^^^^^^^^^^^^^^^^^^^^    

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


Calculations
^^^^^^^^^^^^^^^^^^^^^^^

The arrays of a data container can either be modified directly

.. code-block:: python

    # copy to not overwrite the values
    data_new = {**data}
    data_new['var'] = data['var']/15.3*20.6
    # 'mask' is also just a numpy array, though with boolean values
    data_new['mask'] = np.logical_or(data_new['mask'], data_new['var'] > 7)

It is recommended to use the :meth:`pyLARDA.Transformations.combine` functionality, to not loose the meta-information:

.. code-block:: python

    def correct_Z_bias(data):
        var = data['var'] + h.z2lin(8.)
        return var, data['mask']

    Z_corrected = pyLARDA.Transformations.combine(correct_Z_bias, Z, {})


.. attention::

    Several plotting functions were refactored, consider using the new functions:
    :meth:`pyLARDA.Transformations.plot_timeseries2`, :meth:`pyLARDA.Transformations.plot_timeheight2`, :meth:`pyLARDA.Transformations.plot_scatter2`



Slicing
^^^^^^^^^^^

Subsetting of a data containter is facilitated with the :meth:`pyLARDA.Transformations.slice_container` function.


Conversion and output
^^^^^^^^^^^^^^^^^^^^^^^

Data containers can be easily converted into and `xarray.DataArray` with :meth:`pyLARDA.Transformations.container2DataArray` 
or stored into a netcdf with :meth:`pyLARDA.NcWrite.write_simple_nc`.



function list 
---------------

.. automodule:: pyLARDA.Transformations
   :members:

private functions
------------------

.. automodule:: pyLARDA.Transformations
   :members:
   :private-members:
   :noindex:
