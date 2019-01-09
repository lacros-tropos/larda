#!/usr/bin/python3

import os,sys
import glob
import copy
import re
import datetime
import calendar
import pprint
import functools

import pyLARDA.NcReader as NcReader
import pyLARDA.ParameterInfo as ParameterInfo
#import pyLARDA.DataBuffer as DataBuffer
#import pyLARDA.MeteoReader as MeteoReader
#import pyLARDA.Spec as Spec
import pyLARDA.helpers as helpers
import pyLARDA.Transformations as Transf

import numpy as np
from operator import itemgetter
import collections
import json

import logging
logger = logging.getLogger(__name__)

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
    dt = convert_regex_date_to_dt(
        re.search(datepattern, f).groupdict())
    return dt.strftime("%Y%m%d-%H%M")


def setupreader(paraminfo):
    """

    """

    if paraminfo["ncreader"] == 'timeheight_limrad94':
        reader = NcReader.timeheightreader_rpgfmcw(paraminfo)
    else:
        reader = NcReader.reader(paraminfo)

    return reader


def setup_valid_date_filter(valid_dates):
    def date_filter(e):
        datepair, f = e
        f_b, f_e = [d[:-5] for d in datepair]
        #print(valid_dates, datepair, f_b, f_e)
        #print([(f_b >= valid[0] and f_e <= valid[1]) for valid in valid_dates])
        return any([(f_b >= valid[0] and f_e <= valid[1]) for valid in valid_dates])
    return date_filter



class Connector_remote:
    """ """
    def __init__(self, system, plain_dict, uri):
        print("huhu remote connector here")
        self.system = system
        self.params_list = list(plain_dict['params'].keys())
        print(self.system, self.params_list)
        self.plain_dict = plain_dict
        self.uri = uri


    def collect(self, param, time_interval, *further_intervals):
        """collect the data from a parameter for the given intervals

        Args:
            param (str) identifying the parameter
            time_interval: list of begin and end datetime
            *further_intervals: range, velocity, ...
        """
        raise NotImplemented("not yet implemented") 
        paraminfo = self.system_info["params"][param]
        base_dir = self.system_info['path'][paraminfo['which_path']]["base_dir"]
        print("paraminfo at collect ", paraminfo)
        begin, end = [dt.strftime("%Y%m%d-%H%M") for dt in time_interval]
        # cover all three cases: 1. file only covers first part
        # 2. file covers middle part 3. file covers end
        flist = [e for e in self.filehandler[paraminfo['which_path']] \
                 if (e[0][0] <= begin and e[0][1] > begin) 
                  or (e[0][0] > begin and e[0][1] < end) 
                  or (e[0][0] <= end and e[0][1] >= end)] 
        assert len(flist) > 0, "no files available"

        load_data = setupreader(paraminfo)
        datalist = [load_data(base_dir+e[1], time_interval, *further_intervals) for e in flist]
        #Transf.join(datalist[0], datalist[1])
        data = functools.reduce(Transf.join, datalist)


        return data



class Connector:    
    """connect the data (from the ncfiles/local sources) to larda 
    
    """
    def __init__(self, system, system_info, valid_dates):
        self.system=system
        self.system_info=system_info
        self.valid_dates=valid_dates
        self.params_list = system_info["params"].keys()
        logger.info("params in this connector {} {}".format(self. system, self.params_list))
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
            for root, dirs, files in os.walk(pathinfo['base_dir']):
                #print(root, dirs, len(files), files[:5], files[-5:] )
                current_regex = pathinfo['matching_subdirs']
                abs_filepaths = [root +'/'+ f for f in files if re.search(current_regex, root +'/'+ f)]
    
                all_files += abs_filepaths
                #files = [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]
    
            # remove basedir (not sure if that is a good idea)
            all_files = [p.replace(pathinfo['base_dir'], "./") for p in all_files]
            logger.debug('filelist {} {}'.format(len(all_files), all_files[:10]))
    
            all_files = sorted(all_files)
            dates = [convert_to_datestring(pathinfo["date_in_filename"], f)\
                     for f in all_files]

            if dates:
                if len(dates) > 1:
                    guessed_duration = (datetime.datetime.strptime(dates[-1],'%Y%m%d-%H%M') - 
                        datetime.datetime.strptime(dates[-2],'%Y%m%d-%H%M'))
                else:
                    guessed_duration = datetime.timedelta(hours=24)
                last_data = (
                    datetime.datetime.strptime(dates[-1],'%Y%m%d-%H%M') + guessed_duration
                ).strftime("%Y%m%d-%H%M")
                date_pairs = zip(dates, dates[1:]+[last_data])
            else:
                date_pairs = []
            
            #singlehandler = zip(date_pairs, all_files)
            singlehandler = list(filter(
                setup_valid_date_filter(self.valid_dates), 
                zip(date_pairs, all_files)))
            
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


    def collect(self, param, time_interval, *further_intervals):
        """collect the data from a parameter for the given intervals

        Args:
            param (str) identifying the parameter
            time_interval: list of begin and end datetime
            *further_intervals: range, velocity, ...
        """
        
        paraminfo = self.system_info["params"][param]
        base_dir = self.system_info['path'][paraminfo['which_path']]["base_dir"]
        logger.debug("paraminfo at collect {}".format(paraminfo))
        begin, end = [dt.strftime("%Y%m%d-%H%M") for dt in time_interval]
        # cover all three cases: 1. file only covers first part
        # 2. file covers middle part 3. file covers end
        flist = [e for e in self.filehandler[paraminfo['which_path']] \
                 if (e[0][0] <= begin and e[0][1] > begin) 
                  or (e[0][0] > begin and e[0][1] < end) 
                  or (e[0][0] <= end and e[0][1] >= end)] 
        assert len(flist) > 0, "no files available"

        load_data = setupreader(paraminfo)
        datalist = [load_data(base_dir+e[1], time_interval, *further_intervals) for e in flist]
        #Transf.join(datalist[0], datalist[1])
        data = functools.reduce(Transf.join, datalist)


        return data

    def get_as_plain_dict(self):
        """put the most important information of the connector into a plain dict (for http tranfer)

        .. code::

            {params: {param_name: fileidentifier, ...}, 
             avail: {fileidentifier: {"YYYYMMDD": no_files, ...}, ...}``

        Returns:
            ``dict``
        """
        return {
            'params': {e:self.system_info['params'][e]['which_path'] for e in self.params_list},
            'avail': {k:self.files_per_day(k) for k in self.filehandler.keys()}
        }
    

    def get_matching_files(self, begin_time, end_time):
        """ """
        matching_files=[]
        begin_day=datetime.datetime.utcfromtimestamp(begin_time).date()
        end_day=datetime.datetime.utcfromtimestamp(end_time).date()
        
        for i in range(len(self.datelist)):
            if self.datelist[i]>=begin_day and self.datelist[i]<=end_day :
                matching_files.append(self.filelist[i])
        
        if len(matching_files)==0:
            raise Exception("no files found for "+self.param_info.system_type+" "+self.param_info.variable_name)
        
        return matching_files
        
        
    def files_per_day(self, which_path):
        """replaces ``days_available`` and ``day_available``
        
        Returns:
            ``dict``: ``{'YYYYMMDD': no of files, ...}``
        """
        fh = self.filehandler[which_path]
        groupedby_day = collections.defaultdict(list)
        for d, f in fh:
            groupedby_day[d[0][:8]] += [f]
        no_files_per_day = {k: len(v) for k, v in groupedby_day.items()}
        return no_files_per_day
