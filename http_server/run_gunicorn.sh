#!/bin/bash

gunicorn http_server:app -c gunicorn_config.py 

# ps ax|grep gunicorn 
# pkill gunicorn
# now via upstart and /etc/init/werblarda3.conf

