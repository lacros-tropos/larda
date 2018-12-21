

########################
Data and transformations
########################


Data format
------------
The connector provides the data loaded from the netcdf file as a data contianer (plain python dict with a specific set of keys).
As a rule of thumb the NcReader returns the variable in linear units, the time as unix timestamp and the range in m.
Following keys are currently specified:

====================  ===============================================================
  Key                  Example                            
====================  ===============================================================
 ``ts``                timestamps
 ``rg``                ranges (optional)
 ``vel``               velocity (of Doppler spectrum, optional)
 ``type``              like 'timeseries', 'timeheight', 'spec',...
                                              
 ``var``               the actual data array
 ``mask``              a mask for the data
                                              
 ``paraminfo``         the info dict derived for the parameter config file 
                        (do not mutate)
 ``filename``          the source file
 ``rg_unit``           unit for the range
 ``var_unit``          unit for the variable
 ``var_lims``          limits for the plot
 ``colormap``          colormap to use
====================  ===============================================================



Transformations
---------------
Transformations operate on a single or several data container(s). 

- [x] interpolation (at least for 2d case)
- [x] join two datadicts 
- [ ] unit conversion
- [ ] plotting (aka transform from numeric to graphical data)

    - [x] 2D
    - [ ] specs

- [x] combine two buffers (do calculations like depol, dwr, bias correction)


.. automodule:: pyLARDA.Transformations
   :members:
