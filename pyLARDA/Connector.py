#!/usr/bin/python3

import os, sys, time
import glob
import copy
import re
import datetime
import calendar
import pprint
import functools
import pprint as pprint2
from pathlib import Path

from typing import Callable

import pyLARDA.NcReader as NcReader
import pyLARDA.ParameterInfo as ParameterInfo
#import pyLARDA.DataBuffer as DataBuffer
#import pyLARDA.MeteoReader as MeteoReader
#import pyLARDA.Spec as Spec
import pyLARDA.peakTree as peakTree
import pyLARDA.trace_reader as trace_reader
import pyLARDA.helpers as h
import pyLARDA.Transformations as Transf

import numpy as np
from operator import itemgetter
import collections
import json
import requests, msgpack
from tqdm import tqdm
#import cbor2

import logging
logger = logging.getLogger(__name__)

DATEstrfmt = "%Y%m%d-%H%M%S"

def convert_regex_date_to_dt(re_date): 
    """convert a re_date dict to datetime

    .. warning::

        When using 2 digit years (i.e. RPG) a 20 will
        be added in front

    Args:
        re_date (dict): result of the regex search with keys
    Returns:
        datetime
    """
    l = []
    if len(re_date['year']) == 2:
        re_date['year'] = '20' + re_date['year']
    for k in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        if k in re_date.keys():
            l.append(int(re_date[k]))
    return datetime.datetime(*l)


def convert_to_datestring(datepattern, f):
    """convert the date in a (file-)string to dt

    Args:
        datepatttern: a python regex definition with named groups
        f: the string
    Returns:
        datetime
    """
    try:
        dt = convert_regex_date_to_dt(
            re.search(datepattern, f).groupdict())
    except AttributeError:
        logger.warning(f'No matching data pattern "{datepattern}" in file: "{f}"')
        return -1

    return dt.strftime(DATEstrfmt)


def setupreader(paraminfo) -> Callable:
    """obtain the reader from the paraminfo

    """

    if paraminfo["ncreader"] == 'timeheight_limrad94':
        reader = NcReader.timeheightreader_rpgfmcw(paraminfo)
    elif paraminfo["ncreader"] == 'spec_limrad94':
        reader = NcReader.specreader_rpgfmcw(paraminfo)
    elif paraminfo["ncreader"] == 'spec_rpgpy':
        reader = NcReader.specreader_rpgpy(paraminfo)
    elif paraminfo["ncreader"] == 'spec_kazr':
        reader = NcReader.specreader_kazr(paraminfo)
    elif paraminfo["ncreader"] in ['aux', 'aux_all_ts']:
        reader = NcReader.auxreader(paraminfo)
    elif paraminfo["ncreader"] in ['scan_timeheight', 'scan_time']:
        reader = NcReader.scanreader_mira(paraminfo) 
    elif paraminfo['ncreader'] == 'peakTree':
        reader = peakTree.peakTree_reader(paraminfo)
    elif paraminfo['ncreader'] == 'trace':
        reader = trace_reader.trace_reader(paraminfo)
    elif paraminfo['ncreader'] == 'trace2':
        reader = trace_reader.trace_reader2(paraminfo)
    elif paraminfo["ncreader"] == 'pollyraw':
        reader = NcReader.reader_pollyraw(paraminfo)
    elif paraminfo["ncreader"] == 'mrrpro_spec':
        paraminfo.update({"ncreader": "spec", "compute_velbins":"mrrpro"})
        reader = NcReader.reader(paraminfo)
    elif paraminfo["ncreader"] == "wyoming_sounding_txt":
        reader = NcReader.reader_wyoming_sounding(paraminfo)
    else:
        reader = NcReader.reader(paraminfo)

    return reader


def setup_valid_date_filter(valid_dates) -> Callable:
    """validator function for chunks of valid dates
    
    Args:
        valid_dates: list of [begin, end] in 'YYYYMMDD'
    
    Returns:
        a single argument ('YYYYMMDD-HHMMSS') validator function
    """
    def date_filter(e):
        datepair, f = e
        f_b, f_e = datepair
        #print(valid_dates, datepair, f_b, f_e)
        #print([(f_b >= valid[0] and f_e <= valid[1]) for valid in valid_dates])
        return any([(f_b[:-7] >= valid[0] and f_e[:-7] <= valid[1]) for valid in valid_dates])

    return date_filter


def path_walk(top, topdown = False, followlinks = False):
    """pendant for os.walk
    """
    
    names = list(top.iterdir())

    dirs = [node for node in names if node.is_dir() is True]
    nondirs = [node for node in names if node.is_dir() is False]

    if topdown:
        yield top, dirs, nondirs

    for name in dirs:
        if followlinks or name.is_symlink() is False:
            for x in path_walk(name, topdown, followlinks):
                yield x

    if topdown is not True:
        yield top, dirs, nondirs


def end_1sec_earlier(date):
    dt = datetime.datetime.strptime(date, DATEstrfmt)
    return (dt-datetime.timedelta(seconds=1)).strftime(DATEstrfmt)


def guess_end(dates):
    """estimate the end of a file
    
    Returns:
        list of pairs [begin, end]
    """
    if len(dates) > 1:
        guessed_duration = (datetime.datetime.strptime(dates[-1], DATEstrfmt) - 
            datetime.datetime.strptime(dates[-2], DATEstrfmt))
    else:
        guessed_duration = datetime.timedelta(seconds=(24*60*60)-1)
    # quick fix guessed duration not longer than 24 h
    if guessed_duration >= datetime.timedelta(days=1):
        guessed_duration = datetime.timedelta(seconds=(24*60*60)-1)
    last_d = (
        datetime.datetime.strptime(dates[-1], DATEstrfmt) + guessed_duration
    ).strftime(DATEstrfmt)
    ends = [end_1sec_earlier(d) for d in dates[1:]] + [last_d]
    return list(zip(dates, ends))


class Connector_remote:
    """connect the data (from the a remote source) to larda

    Args:
        camp_name (str): campaign name
        system (str): system identifier
        plain_dict (dict): connector meta info
        uri (str): address of the remote source
    """
    def __init__(self, camp_name, system, plain_dict, uri):
        self.camp_name = camp_name
        self.system = system
        self.params_list = list(plain_dict['params'].keys())
        print(self.system, self.params_list)
        self.plain_dict = plain_dict
        self.uri = uri

    def collect(self, param, time_interval, *further_intervals, **kwargs) -> dict:
        """collect the data from a parameter for the given intervals

        Args:
            param (str) identifying the parameter
            time_interval: list of begin and end datetime
            *further_intervals: range, velocity, ...
            **interp_rg_join: interpolate range during join

        Returns:
            data_container
        """
        resp_format = 'msgpack'
        interval = ["-".join([str(h.dt_to_ts(dt)) for dt in time_interval])]
        interval += ["-".join([str(i) for i in pair]) for pair in further_intervals]
        stream = True if resp_format == "msgpack" else False
        params = {"interval": ','.join(interval), 'rformat': resp_format}
        params.update(kwargs)
        resp = requests.get(self.uri + '/api/{}/{}/{}'.format(self.camp_name, self.system, param),
                            params=params, stream=stream)
        logger.debug("fetching data from: {}".format(resp.url))
        if resp_format == "msgpack":
            block_size = 1024
            pbar = tqdm(unit="B", total=(int(resp.headers.get('content-length', 0))//block_size)*block_size, unit_divisor=1024, unit_scale=True)
            content = bytearray()
            for data in resp.iter_content(block_size):
                content.extend(data)
                pbar.update(len(data))
        
        if resp.status_code != 200:
            if resp_format == "msgpack":
                print("Error at Backend")
                print(content.decode("unicode_escape"))
            else:
                print(resp.json())
            raise ConnectionError("bad status code of response {}".format(resp.status_code))

        starttime = time.time()
        # if resp_format == 'bin':
        #     data_container = cbor2.loads(resp.content)
        if resp_format == 'msgpack':
            logger.info("msgpack version {}".format(msgpack.version))
            if msgpack.version[0] < 1:
                data_container = msgpack.loads(content, encoding='utf-8')
            else:
                data_container = msgpack.loads(content, strict_map_key=False)
        elif resp_format == 'json':
            data_container = resp.json()

        #print("{:5.3f}s decode data".format(time.time() - starttime))
        starttime = time.time()
        for k in ['ts', 'rg', 'vel', 'var', 'mask', 'vel_ch2', 'vel_ch3']:
            if k in data_container and type(data_container[k]) == list:
                data_container[k] = np.array(data_container[k])
        logger.info("loaded data container from remote: {}".format(data_container.keys()))
        #print("{:5.3f}s converted to np arrays".format(time.time() - starttime))
        return data_container


    def description(self, param):
        """get the description str"""
        resp = requests.get(self.uri + '/description/{}/{}/{}'.format(self.camp_name, self.system, param))
        if resp.status_code != 200:
            raise ConnectionError("bad status code of response {}".format(resp.status_code))

        logger.warning(resp.text)
        return resp.text 


    def get_as_plain_dict(self) -> dict:
        """put the most important information of the connector into a plain dict (for http tranfer)"""

        return self.plain_dict

class Connector:
    """connect the data (from the ncfiles/local sources) to larda


    Args:
        system (str): system identifier
        system_info (dict): dict info loaded from toml
        valid_dates (list of lists): list of begin and end datetime
        description_dir (optional): dir with the description rst
    """
    def __init__(self, system, system_info, valid_dates, description_dir=None):
        self.system = system
        self.system_info = system_info
        self.valid_dates = valid_dates
        self.params_list = list(system_info["params"].keys())
        self.description_dir = description_dir
        logger.info("params in this connector {} {}".format(self.system, self.params_list))
        logger.debug('connector.system_info {}'.format(system_info))

    def __str__(self):
        s = "connector for system {} \ncontains parameters: ".format(self.system)
        s += " ".join(self.params_list)
        return s

    def build_filehandler(self):
        """scrape the directories and build the filehandler
        """
        pathdict = self.system_info['path']

        filehandler = {}
        for key, pathinfo in pathdict.items():
            all_files = []
            current_regex = pathinfo['matching_subdirs'] if 'matching_subdirs' in pathinfo else ''

            # 1. match the names and subdirs with regex
            for root, _, files in path_walk(Path(pathinfo['base_dir'])):
                #print('walk ', root, dirs, len(list(files)))
                abs_filepaths = [f for f in files if re.search(current_regex, str(f))]
                logger.debug("valid_files {} {}".format(root, [f for f in files if re.search(current_regex, str(f))]))
                #print("skipped_files {} {}".format(root, [f for f in files if not re.search(current_regex, str(f))]))
                all_files += abs_filepaths
                #files = [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]
    
            # remove basedir (not sure if that is a good idea)
            all_files = [str(p).replace(pathinfo['base_dir'], "./") for p in all_files]
            #logger.debug('filelist {} {}'.format(len(all_files), all_files[:10]))

            # 2. extract the dates with another regex
            dates = [convert_to_datestring(pathinfo["date_in_filename"], str(f))\
                     for f in all_files]
            all_files = [f for _, f in sorted(zip(dates, all_files), key=lambda pair: pair[0])]
            dates = sorted(dates)

            # 3. estimate the duration a file covers
            date_pairs = guess_end(dates) if dates else []
            
            # 4. validate with the durations
            valid_date_filter = setup_valid_date_filter(self.valid_dates)
            singlehandler = list(filter(
                valid_date_filter, 
                list(zip(date_pairs, all_files))))
            
            filehandler[key] = singlehandler
        #pprint.pprint(filehandler)
        self.filehandler = filehandler 


    def save_filehandler(self, path, camp_name):
        """save the filehandler to json file"""
        savename = 'connector_{}.json'.format(self.system)
        pretty = {'indent': 2, 'sort_keys':True}
        #pretty = {}

        if not os.path.isdir(path+'/'+camp_name):
            os.makedirs(path+'/'+camp_name)

        with open(path+'/'+camp_name+'/'+savename, 'w') as outfile:
                json.dump(self.filehandler, outfile, **pretty)
                logger.info('saved connector to {}/{}/{}'.format(path,camp_name,savename))

    def load_filehandler(self, path, camp_name):
        """load the filehandler from the json file"""
        filename = "connector_{}.json".format(self.system)
        with open(path+'/'+camp_name+'/'+filename) as json_data:
                self.filehandler = json.load(json_data)

    def collect(self, param, time_interval, *further_intervals, **kwargs) -> dict:
        """collect the data from a parameter for the given intervals

        Args:
            param (str) identifying the parameter
            time_interval: list of begin and end datetime
            *further_intervals: range, velocity, ...
            **interp_rg_join: interpolate range during join

        Returns:
            data_container
        """
        
        paraminfo = self.system_info["params"][param]
        if 'interp_rg_join' not in paraminfo:
            # default value
            paraminfo['interp_rg_join'] = False
        if 'interp_rg_join' in kwargs:
            paraminfo['interp_rg_join'] = kwargs['interp_rg_join']
        base_dir = self.system_info['path'][paraminfo['which_path']]["base_dir"]
        logger.debug("paraminfo at collect {}".format(paraminfo))
        if len(time_interval) == 2:
            begin, end = [dt.strftime(DATEstrfmt) for dt in time_interval]
            # cover all three cases: 1. file only covers first part
            # 2. file covers middle part 3. file covers end
            #print(begin, end)
            flist = [e for e in self.filehandler[paraminfo['which_path']] \
                     if (e[0][0] <= begin < e[0][1])
                     or (e[0][0] > begin and e[0][1] < end)
                     or (e[0][0] <= end <= e[0][1])]
            assert len(flist) > 0, "no files available"
        elif len(time_interval) == 1:
            begin = time_interval[0].strftime(DATEstrfmt)
            flist = [e for e in self.filehandler[paraminfo['which_path']] if e[0][0] <= begin < e[0][1]]
            assert len(flist) == 1, "flist too long or too short: {}".format(len(flist))

        #[print(e, (e[0][0] <= begin and e[0][1] > begin), (e[0][0] > begin and e[0][1] < end), (e[0][0] <= end and e[0][1] >= end)) for e in flist]

        load_data = setupreader(paraminfo)
        datalist = [load_data(base_dir + e[1], time_interval, *further_intervals) for e in flist]
        # [print(e.keys) if e != None else print("NONE!") for e in datalist]
        # reader returns none, if it detects no data prior to begin
        # now these none values are filtered from the list
        assert len(datalist) > 0, 'No data found for parameter: {}'.format(param)
        datalist = list(filter(lambda x: x != None, datalist))
        #Transf.join(datalist[0], datalist[1])
        data = functools.reduce(Transf.join, datalist)

        return data


    def description(self, param) -> str:
        paraminfo = self.system_info["params"][param]
        #print('connector local paraminfo: ' + paraminfo['variable_name'])

        # Prints the nicely formatted dictionary
        # this is the python pprint function, not the larda.helpers function
        pp = pprint2.PrettyPrinter(indent=4)
        logger.info(pp.pformat(paraminfo))

        if 'description_file' not in paraminfo:
            return 'no description file defined in config'
        if self.description_dir == None:
            return 'description dir not set'
        
        description_file = self.description_dir / paraminfo['description_file']
        logger.info('load description file {}'.format(description_file))

        with open(description_file, 'r', encoding="utf-8") as f:
            descr = f.read()
        descr = "\n"+descr+"\n"
        logger.warning(descr)
        return descr        

    def get_as_plain_dict(self) -> dict:
        """put the most important information of the connector into a plain dict (for http tranfer)

        Returns:
            connector information

            .. code::

                {params: {param_name: fileidentifier, ...},
                avail: {fileidentifier: {"YYYYMMDD": no_files, ...}, ...}
        """
        return {
            'params': {e: self.system_info['params'][e]['which_path'] for e in self.params_list},
            'avail': {k: self.files_per_day(k) for k in self.filehandler.keys()}
        }

    def files_per_day(self, which_path) -> dict:
        """replaces ``days_available`` and ``day_available``

        Returns:
            dict with days and no of files

            .. code::

                {'YYYYMMDD': no of files, ...}
        """
        fh = self.filehandler[which_path]
        groupedby_day = collections.defaultdict(list)
        for d, f in fh:
            groupedby_day[d[0][:8]] += [f]
        no_files_per_day = {k: len(v) for k, v in groupedby_day.items()}
        return no_files_per_day
