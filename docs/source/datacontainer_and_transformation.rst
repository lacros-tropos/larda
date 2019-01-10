
####################################
Data container and transformations
####################################


Data container format
---------------------
The connector provides the data loaded from the netcdf file as a data contianer (plain python ``dict`` with a specific set of keys).
As a rule of thumb the :meth:`pyLARDA.NcReader` returns the variable in linear units, the time as unix timestamp and the range in m.
Following keys are currently specified:

====================  =========================================================================
  Key                  Example                            
====================  =========================================================================
 ``ts``                timestamps
 ``rg``                ranges (optional)
 ``vel``               velocity (of Doppler spectrum, optional)
 ``dimlabel``          eg ``['time']``, ``['time', 'range']``, ``['time', 'range', 'vel']``
                                              
 ``var``               the actual data array
 ``mask``              a mask for the data
                                              
 ``paraminfo``         the info dict derived for the parameter config file (do not mutate)
 ``filename``          the source file
 ``rg_unit``           unit for the range
 ``var_unit``          unit for the variable
 ``var_lims``          limits for the plot
 ``colormap``          colormap to use
====================  =========================================================================



Transformations
---------------
Transformations operate on a single or several data container(s). 


.. code:: python

    larda=pyLARDA.LARDA().connect_local('lacros_dacapo', build_lists=True)
    MIRA_Zg=larda.read("MIRA","Zg", [dt_begin, dt_end], [0, 4000])
    MIRA_Zg['var_lims'] = [-50,0]
    fig, ax = pyLARDA.Transformations.plot2d(MIRA_Zg, range_interval=[500, 3000],
                                             z_converter='lin2z')
    fig.savefig(‘MIRA_Z.png’, dpi=250)
                                                                 
    # or for interpolation
    interpolated_container = pyLARDA.Transformations.interp2d(MIRA_Zg, new_times, new_ranges)



.. automodule:: pyLARDA.Transformations
   :members:
