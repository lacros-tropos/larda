####################################
Setup remote date source
####################################

Besides the local pyLARDA variant, a remote data source can be set up.
The data_containers for a requested time/range interval are then loaded from the server
via http and compressed with msgpack_.

.. _msgpack: https://msgpack.org/

.. code-block:: python

    larda = pyLARDA.LARDA('remote', uri='http://<the server>')




Additional requirements
-----------------------

.. code-block:: python

    flask
    flask_cors
    cbor
    gunicorn



Setup
-----

- adapt the ``http_server/gunicorn_config.py`` to your needs
- configure gunicorn as a upstart or systemctl service 
- configure apache/nginx as a reverse proxy to the port of gunicorn
- if required populate the ``http_server/public`` folder with the weblarda frontend
- setup the ``ListCollector.py`` to be run by a cronjob to automatically update the connectordump