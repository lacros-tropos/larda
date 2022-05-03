####################################
Setup remote date source
####################################

Besides the local pyLARDA variant, a remote data source can be set up.
The data_containers for a requested time/range interval are then loaded from the server
via http and compressed with msgpack_.

.. _msgpack: https://msgpack.org/


Additional requirements
-----------------------

.. code-block:: python

    flask
    flask_cors
    cbor
    gunicorn

.. attention::

    It is strongly recommended to use an virtural environment. ``python3 -m venv larda-env``


Setup
-----

gunicorn
^^^^^^^^

Adapt the ``larda/http_server/gunicorn_config.py`` to your needs. gunicorn will the backend at the specified port (default: 7979).
To run gunicorn permanently a service has to be set up. On older operating systems upstart might be available, newer ubuntu versions use systemd.

upstart
^^^^^^^

Put the config into the upstart dir (usually ``/etc/init/weblarda3.conf``)
Likely you will have to adapt the location of the python executable.

.. code-block:: none

    description "weblarda3"

    start on started sshd
    stop on shutdown

    chdir /home/larda3/larda/http_server
    exec /home/larda3/miniconda3/bin/gunicorn http_server:app -c gunicorn_config.py

The server can be started with ``start weblarda3``


systemd
^^^^^^^

The configuration is usually located in ``/etc/systemd/system/weblarda3.service``

.. code-block:: none

    [Unit]
    Description=weblarda3
    After=network.target

    [Service]
    User=larda3
    #Restart=on-failure
    Restart=always
    RuntimeMaxSec=10800
    WorkingDirectory=/home/larda3/larda/http_server
    ExecStart=/home/larda3/larda-env/bin/gunicorn -c gunicorn_config.py http_server:app

    [Install]
    WantedBy=multi-user.target

``systemctl daemon-reload``  ``systemctl enable weblarda3`` ``systemctl start weblarda3``

.. hint::

    Systems running SELinux might require a modification of the type enforcement rules


- Now check with ``curl localhost:7979/api/``. The response should be a list of available campaigns.
- to make the gunicorn server at port 7979 accessible by the outside, a proxy server (apache/nginx) has to be set up.

apache
^^^^^^^
For example the apache site configuration might look like

.. code-block:: none

    <VirtualHost *:80>
        ServerName larda.tropos.de
        DocumentRoot /lacroshome/larda/www/
        ErrorLog logs/larda_error_log
        CustomLog logs/larda_custom_log common
        
        ProxyPreserveHost On
        ProxyPass /larda3/ http://127.0.0.1:7979/ timeout=600 Keepalive=On
        ProxyPassReverse /larda3/ http://larda.tropos.de/larda3/ timeout=600
    </VirtualHost>

When in doubt, contact your sysadmin.


frontend
^^^^^^^^

The files for the larda frontend (data availability overview and explorer) can be placed in ``http_server/public``.

.. note::

    TODO: make the frontend files downloadable


cronjob
^^^^^^^

The connectordump should be updated regularly with a cronjob calling ``ListCollector.py``, to speed up
data loading.


Finally, the remote can be used:

.. code-block:: python

    larda = pyLARDA.LARDA('remote', uri='http://<the server>')
