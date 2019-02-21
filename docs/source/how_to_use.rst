
######################
How to use larda
######################

Initialize
----------

.. code-block:: python

    #!/usr/bin/python3
    import sys
    sys.path.append('<path to local larda directory>')
    import pyLARDA
    import pyLARDA.helpers as h

    # optionally configure the logging
    # StreamHandler will print to console
    import logging
    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

    # init larda
    # either using local data
    larda=pyLARDA.LARDA()
    # or loading data via http from backend
    larda = pyLARDA.LARDA('remote', uri='<url to larda remote backend>')
    print("available campaigns ", larda.campaign_list)

    # select a campaign
    larda.connect('lacros_dacapo')


Load data
---------

Get the data container for a certain system and parameter.


.. code-block:: python

    improt datetime
    # Initialize larda as described above

    begin_dt = datetime.datetime(2018, 12, 18, 6, 0)
    end_dt = datetime.datetime(2018, 12, 18, 11, 0)

    MIRA_Zg = larda.read("MIRA", "Zg", [begin_dt, end_dt], [0, 'max'])
    #
    shaun_vel=larda.read("SHAUN", "VEL", [begin_dt, end_dt], [0, 'max'])


Simple plot
-----------

.. code-block:: python
        
    begin_dt = datetime.datetime(2018, 12, 18, 6, 0)
    end_dt = datetime.datetime(2018, 12, 18, 11, 0)
    CLOUDNET_Z = larda.read("CLOUDNET", "Z", [begin_dt, end_dt], [0, 'max'])
    fig, ax = pyLARDA.Transformations.plot_timeheight(
        CLOUDNET_Z, range_interval=[300, 12000], z_converter='lin2z')
    fig.savefig('cloudnet_Z.png', dpi=250)

    CLOUDNET_class = larda.read("CLOUDNET", "CLASS", [begin_dt, end_dt], [0, 'max'])
    fig, ax = pyLARDA.Transformations.plot_timeheight(
        CLOUDNET_class, range_interval=[300, 12000])
    fig.savefig('cloudnet_class.png', dpi=250)


.. image:: ../plots_how_to_use/cloudnet_Z.png
    :width: 400px
    :align: center

.. image:: ../plots_how_to_use/cloudnet_class.png
    :width: 400px
    :align: center



Modify plot appareance
----------------------

.. code-block:: python

    begin_dt=datetime.datetime(2019,2,4,0,1)
    end_dt=datetime.datetime(2019,2,5,20)
    plot_range = [50, 6500]

    attbsc1064 = larda.read("POLLY","attbsc1064",[begin_dt,end_dt],[0,8000])
    attbsc1064['colormap'] = 'jet'
    fig, ax = pyLARDA.Transformations.plot_timeheight(
            attbsc1064, range_interval=plot_range, fig_size=[20,5.7], z_converter="log")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d.%m. %H:%M'))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,]))
    fig.savefig('polly_bsc1064.png', dpi=250)


.. image:: ../plots_how_to_use/polly_bsc1064.png
    :width: 600px
    :align: center



Scatter plot
------------

.. code-block:: python

    begin_dt = datetime.datetime(2018, 12, 6, 0, 0, 0)
    end_dt   = datetime.datetime(2018, 12, 6, 0, 30, 0)

    # load the reflectivity data
    MIRA_Z = larda.read("CLOUDNET", "Z", [begin_dt, end_dt], [0, 'max'])
    LIMRAD94_Z = larda.read("CLOUDNET_LIMRAD", "Z", [begin_dt, end_dt], [0, 'max'])

    LIMRAD94_Z_interp = pyLARDA.Transformations.interpolate2d(LIMRAD94_Z, 
                                            new_time=MIRA_Z['ts'], new_range=MIRA_Z['rg'])
    fig, ax = pyLARDA.Transformations.plot_scatter(MIRA_Z, LIMRAD94_Z_interp, var_lim=[-75, 20],
                                            x_lim = [-50, 10], y_lim = [-50, 10],
                                            custom_offset_lines=5.0, z_converter='lin2z')
    fig.savefig('scatter_mira_limrad_Z.png', dpi=250)


.. image:: ../plots_how_to_use/scatter_mira_limrad_Z.png
    :width: 350px
    :align: center