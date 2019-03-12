
####################################
peakTree extension
####################################

Setup
-----
Optionally, for faster reading (building the trees from the netcdf file) a cython extension can be used.
This extension has to be complied:

.. code-block:: python

    python3 setup.py build_ext --inplace


Load and plot
-------------
Loading the data works similar to the standard larda, only the third dimension of
the data_container is a ``dict`` of nodes instead a further array dimension.
To plot these data into a time height crosssection, a parameter (reflectivity, velocity, width, ...)
and a node index have to be choosen.

.. code-block:: python

    larda=pyLARDA.LARDA().connect('lacros_dacapo')

    begin_dt = datetime.datetime(2019, 2, 19, 4, 0)
    end_dt = datetime.datetime(2019, 2, 19, 5, 30)
    trees=larda.read("peakTree", "tree", [begin_dt, end_dt], [0, 10000])

    # the tree_to_timeheight selects a 2d array from the ['time', 'range', 'dict'] data container
    # here just the total number of nodes is returned
    no_nodes = pyLARDA.peakTree.tree_to_timeheight(trees, 'no_nodes')
    no_nodes['name'] = 'no. nodes'
    no_nodes['var_unit'] = ''
    # plot_no_nodes provides a wrapper with special colormaps
    fig, ax = pyLARDA.peakTree.plot_no_nodes(no_nodes, fig_size=[13, 5.7])
    fig.savefig("{}_no_nodes.png".format(begin_dt.strftime("%Y%m%d-%H%M")), dpi=250)

    # plot Reflectivity of node 0
    z_node0 = pyLARDA.peakTree.tree_to_timeheight(trees, 'z', sel_node=0)
    z_node0['name'] = 'Reflectivity'
    z_node0['var_unit'] = 'dBZ'
    fig, ax = Transf.plot_timeheight(z_node0, fig_size=[13, 5.7])
    fig.savefig("{}_Z_node0.png".format(begin_dt.strftime("%Y%m%d-%H%M")), dpi=250)


select nodes based on rule
--------------------------
Special features can easily be detected by selection rules. An example is already provided with 
:py:func:`pyLARDA.peakTree.select_liquid_node`. The found index is returned as a 2d array inside a data container.

.. code-block:: python

    # select the indices by a rule
    # here: node['z'] < -20 and abs(node['v']) < 0.3
    selected_index = pyLARDA.peakTree.select_liquid_node(trees)

    # plot the selected index
    fig, ax = pyLARDA.peakTree.plot_sel_index(selected_index, fig_size=[13, 5.7])
    fig.savefig("{}_ind_node_liq_wide.png".format(begin_dt.strftime("%Y%m%d-%H%M")), dpi=250)

    # extract a 2d array of the parameters
    liquid_z = pyLARDA.peakTree.tree_to_timeheight(trees, 'z', sel_node=selected_index['var'])
    liquid_z['name'] = 'Reflectivity'
    liquid_z['var_unit'] = 'dBZ'

    # and plot
    fig, ax = Transf.plot_timeheight(liquid_z, fig_size=[13, 5.7])
    fig.savefig("{}_Z_liq_wide.png".format(begin_dt.strftime("%Y%m%d-%H%M")), dpi=250)


print single tree
-----------------

.. code-block:: python

    single_tree = pyLARDA.Transformations.slice_container(
        trees, value={'time': [h.dt_to_ts(datetime.datetime(2019,2,19,4,16,30))], 
                    'range': [2100]})
    pyLARDA.peakTree.print_tree(single_tree)


Standalone
----------
The :py:mod:`pyLARDA.peakTree` module can be used without the main part of pyLARDA, when loading the file manually.
For convenience one might add the :py:mod:`pyLARDA.Transformations`.


.. code-block:: python

    import pyLARDA.peakTree
    import pyLARDA.helpers as h
    import pyLARDA.Transformations as Transf

    # configuration which is normally provided by the 
    # connector/toml file
    paraminfo = {'time_variable': "timestamp",
                'range_variable': 'range',
                'time_conversion': "unix",
                'range_conversion': "none",
                'var_conversion': 'none',
                'system': "peaktree_peako",
                'var_unit': "tree",
                'rg_unit': "m",
                'paramkey': "trees",
                'colormap': "gist",
                'var_lims': [-99, -99],
                'altitude': 181,
                'ncreader': 'peakTree'}

    rdr = pyLARDA.peakTree.peakTree_reader(paraminfo)
    filename = "../20140202_1600_kazrbaeccpeako_peakTree.nc4"

    trees = rdr(filename, [begin_dt, end_dt], [200, 3000])
    # trees is then the standard data_container of dim ['time', 'range', 'dict']

doc-strings
---------------

.. automodule:: pyLARDA.peakTree
   :members:
