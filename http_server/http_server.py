#!/usr/bin/python3

import sys, os
# just needed to find pyLARDA from this location
sys.path.append('../')

import datetime
import pprint
import toml
import time, json
#import cbor    #library for binary data transfer
import msgpack #try the messagepack binary format 

import pyLARDA
import pyLARDA.helpers as h
from flask import Flask, jsonify, request, Response, send_file, redirect
from flask_cors import CORS
#from flask_compress import Compress
import traceback

import numpy as np

import logging
import logging.handlers

app = Flask(__name__, static_url_path='', static_folder='public')
CORS(app)

#COMPRESS_MIMETYPES = ['application/json']
#COMPRESS_LEVEL = 18
#Compress(app)


app.logger.setLevel(logging.DEBUG)
log_larda = logging.getLogger('pyLARDA')
log_larda.setLevel(logging.INFO)
log_w = logging.getLogger('werkzeug')
log_w.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(process)d] %(name)s %(levelname)s: %(message)s')

#can be used when migrated to pyLARDA
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

print(logging.Logger.manager.loggerDict.keys())
if 'gunicorn' in logging.Logger.manager.loggerDict.keys():
    print('found gunicorn logger')
    gunicorn_e = logging.getLogger('gunicorn.error')
    print(gunicorn_e.name, gunicorn_e.level, gunicorn_e.handlers)
    gunicorn_a = logging.getLogger('gunicorn.access')
    gunicorn_a.setLevel(logging.DEBUG)
    print(gunicorn_a.name, gunicorn_a.level, gunicorn_a.handlers)
    for handler in gunicorn_a.handlers:
        print(handler, handler.level)
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        log_w.addHandler(handler)
        log_larda.addHandler(handler)
else:
    print('no gunicorn logger')
    fh = logging.handlers.RotatingFileHandler("larda_from_flask.log", maxBytes=10e6, backupCount=3)
    fh.setFormatter(formatter)

    app.logger.addHandler(fh)
    log_w.addHandler(fh)
    log_larda.addHandler(fh)


@app.errorhandler(500)
def page_not_found(error):
    exc_info = sys.exc_info()
    return Response(json.dumps(traceback.format_exc()), status=500, mimetype='application/json')


@app.route('/')
def root():
    #return app.send_static_file('index.html')
    # necessary to return location header
    resp = send_file('public/index.html')
    #resp.headers["Location"] = '/'
    return resp
    #return redirect('/larda3/data_avail')


@app.route('/explorer/<camp>')
def explorer(camp):
    #return app.send_static_file('index.html')
    # necessary to return location header
    resp = send_file('public/explorer.html')
    #resp.headers["Location"] = '/'
    return resp
    #return redirect('/larda3/data_avail')

@app.route('/api/', methods=['GET'])
def api_entry():
    """

    Returns:
        campaign list
    """
    larda = pyLARDA.LARDA()
    return jsonify(campaign_list=larda.campaign_list)


@app.route('/api/<campaign_name>/', methods=['GET'])
def get_campaign_info(campaign_name):
    """

    Returns:
        json object with all the campaign info
    """
    campaign_info = {}
    larda=pyLARDA.LARDA().connect(campaign_name, build_lists=False)

    campaign_info['config_file'] = larda.camp.info_dict
    #print("Parameters in stock: ",[(k, larda.connectors[k].params_list) for k in larda.connectors.keys()])
    campaign_info['info_text'] = larda.camp.INFO_TEXT
    
    campaign_info['connectors'] = {system:conn.get_as_plain_dict() for system, conn in larda.connectors.items()}

    return jsonify(**campaign_info)


@app.route('/api/<campaign_name>/<system>/<param>', methods=['GET'])
def get_param(campaign_name, system, param):
    """ """
    app.logger.info("got request for {} {} {}".format(campaign_name, system, param))
    starttime = time.time()
    larda=pyLARDA.LARDA().connect(campaign_name, build_lists=False)
    app.logger.debug("{:5.3f}s load larda".format(time.time() - starttime))

    if "rformat" in request.args and request.args['rformat'] == 'bin':
        rformat = 'bin'
    elif "rformat" in request.args and request.args['rformat'] == 'msgpack':
        rformat = 'msgpack'
    else:
        rformat = 'json'
    intervals = request.args.get('interval').split(',') 
    time_interval = [h.ts_to_dt(float(t)) for t in intervals[0].split('-')]
    further_slices = [[float(e) if e not in ['max'] else e for e in s.split('-')] for s in intervals[1:]]
    
    app.logger.warning('request.args {}'.format(dict(request.args)))
    app.logger.info("time request {}".format(time_interval))
    starttime = time.time()
    data_container = larda.read(system, param, time_interval, *further_slices, **dict(request.args))
    app.logger.debug("{:5.3f}s read data".format(time.time() - starttime))
    starttime = time.time()
    #for k in data_container.keys():
    #    app.logger.warning(f'{k} {type(data_container[k])}')
    for k in ['ts', 'rg', 'vel', 'var', 'mask', 'vel_ch2', 'vel_ch3']:
        if k in data_container and hasattr(data_container[k], 'tolist'):
            if data_container[k].dtype is not np.dtype('object'):
                data_container[k][~np.isfinite(data_container[k])] = 0
            data_container[k] = data_container[k].tolist()
        #if k in data_container:
        #    app.logger.warning(f'{k} {type(data_container[k])}')
    #for k in data_container.keys():
    #    app.logger.warning(f'{k} {type(data_container[k])}')
    app.logger.debug("{:5.3f}s convert data".format(time.time() - starttime))

    #import io
    #test_datacont = {**data_container}
    #for k in ['ts', 'rg', 'var', 'mask']:
    #    if k in test_datacont:
    #        print(k,type(test_datacont[k]))
    #        start = time.time()
    #        test_datacont[k][~np.isfinite(test_datacont[k])] = 0
    #        memfile = io.BytesIO()
    #        np.save(memfile, test_datacont[k])
    #        memfile.seek(0)
    #        test_datacont[k] = memfile.read().decode('latin-1')
    #resp = Response(json.dumps(test_datacont), status=200, mimetype='application/json')

    starttime = time.time()
    #if rformat == 'bin':
    #    resp = Response(cbor.dumps(data_container), status=200, mimetype='application/cbor')
    if rformat == 'msgpack':
        resp = Response(msgpack.packb(data_container), status=200, mimetype='application/msgpack')
    elif rformat == 'json':
        resp = Response(json.dumps(data_container), status=200, mimetype='application/json')
    app.logger.debug("{:5.3f}s dumps {}".format(time.time() - starttime, rformat))

    #for some reason the manual response is faster...
    #return jsonify(data_container)
    return resp

@app.route('/description/<campaign_name>/<system>/<parameter>', methods=['GET'])
def get_descript(campaign_name, system, parameter):
    """ """
    app.logger.info("got request fori description {} {} {}".format(campaign_name, system, parameter))

    larda=pyLARDA.LARDA().connect(campaign_name, build_lists=False)

    #if "rformat" in request.args and request.args['rformat'] == 'bin':
    #    rformat = 'bin'
    #elif "rformat" in request.args and request.args['rformat'] == 'msgpack':
    #    rformat = 'msgpack'
    #else:
    #    rformat = 'json'
    
    app.logger.warning('request.args {}'.format(dict(request.args)))
    text_string = larda.description(system, parameter)

    #if rformat == 'bin':
    #    resp = Response(cbor.dumps(data_container), status=200, mimetype='application/cbor')
    #elif rformat == 'msgpack':
    #    resp = Response(msgpack.packb(data_container), status=200, mimetype='application/msgpack')
    #elif rformat == 'json':
    #    resp = Response(json.dumps(data_container), status=200, mimetype='application/json')
    resp = Response(text_string.encode('utf-8'), status=200, mimetype='text/plain')

    return resp

@app.route('/peakTree')
def peakTree():
    return app.send_static_file('peakTreeVis.html')
