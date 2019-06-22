#!/usr/bin/python3


import pyLARDA.Connector as Connector
import pyLARDA.ParameterInfo as ParameterInfo

import datetime, os, calendar, copy, time
import numpy as np
import csv
import logging
import toml
import requests
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

class LARDA :
    """init a new larda instance

    Args:
        data_source (str, optional): either ``'local'`` or ``'remote'``
        uri: link to backend

    """
    def __init__(self, data_source='local', uri=None):
        if data_source == 'local':
            self.data_source = 'local'
            self.camp = LARDA_campaign(ROOT_DIR + "/../../larda-cfg/", 'campaigns.toml')
            self.campaign_list = self.camp.get_campaign_list()
        elif data_source == 'remote':
            self.data_source = 'remote'
            self.uri = uri
            resp = requests.get(self.uri + '/api/')
            print(resp.json())
            self.campaign_list = resp.json()['campaign_list']

    def connect(self, *args, **kwargs):
        "switch to decide whether connect to local or remote data source"
        if self.data_source == 'local':
            return self.connect_local(*args, **kwargs)
        elif self.data_source == 'remote':
            return self.connect_remote(*args, **kwargs)


    def connect_local(self, camp_name, build_lists=True):
        """built the connector list for the specified campaign (only valid systems are considered)
        the connectors are instances of the Connector.Connector Class

        NEW: one connector per system
        then the params are parts of this system

        Args:
            camp_name (str): name of campaign as defined in ``campaigns.toml``
            build_lists (Bool, optional): Flag to build the filelists or not (with many files this may take some time)
        """

        self.connectors={}

        logger.info("campaign list " + ' '.join(self.camp.get_campaign_list()))
        logger.info("camp_name set {}".format(camp_name))

        description_dir = ROOT_DIR + "/../../larda-description/"

        self.camp.assign_campaign(camp_name)
        config_file=self.camp.CONFIGURATION_FILE
        cinfo_hand_down = {'coordinates': self.camp.COORDINATES,
                           'altitude': self.camp.ALTITUDE,
                           'location': self.camp.LOCATION,
                           'mira_azi_zero': self.camp.info_dict['mira_azi_zero']}
        logger.debug("config file {}".format(config_file))
        
        paraminformation = ParameterInfo.ParameterInfo(
            self.camp.config_dir + config_file, 
            cinfo_hand_down=cinfo_hand_down)
        starttime = time.time()
        logger.info("camp.VALID_SYSTEMS {}".format(self.camp.VALID_SYSTEMS))
        
        #if camp_name == 'LACROS_at_Leipzig':
        #    build_lists = False
        # build the filelists or load them from json
        for system, systeminfo in paraminformation.iterate_systems():
            logger.debug('current parameter {} {}'.format(system, systeminfo))

            conn = Connector.Connector(system, systeminfo,
                                       self.camp.VALID_DATES,
                                       description_dir=description_dir)
            
            if build_lists:
                conn.build_filehandler()
                conn.save_filehandler(self.camp.info_dict['connectordump'], camp_name)

            else:
                #load lists
                conn.load_filehandler(self.camp.info_dict['connectordump'], camp_name)
            
            if system in self.camp.VALID_SYSTEMS:
                self.connectors[system] = conn
            else:
                logger.warning("{} not in valid systems".format(system))

        logger.debug('Time for generating connectors {} s'.format(time.time() - starttime))
        #print "Availability array: ", self.array_avail(2014, 2)
        logger.info("systems {}".format(self.connectors.keys()))
        logger.info("Parameters in stock: {}".format([(k, self.connectors[k].params_list) for k in self.connectors.keys()]))
        return self
        

    def connect_remote(self, camp_name, **kwargs):
        logger.info("connect_remote {}".format(camp_name))
        resp = requests.get(self.uri + '/api/{}/'.format(camp_name))
        #print(resp.json())

        self.connectors = {}
        for k, c in resp.json()['connectors'].items():
            self.connectors[k] = Connector.Connector_remote(camp_name, k, c, self.uri)

        return self


    def read(self, system, parameter, time_interval, *further_slices, **kwargs):
        """
        Args:
            system (str): identifier for the system
            parameter (str): choosen param
            time_interval: ``[dt, dt]`` time interval, or [dt] one time
            *further_slices: range, vel,.. ``[0, max]`` or [3000]

        Returns:
            the dictionary with data
        """
        data = self.connectors[system].collect(parameter, time_interval, *further_slices, **kwargs) 

        return data

    def description(self, system, parameter):
        """
        Args:
            system (str): identifier for the system
            parameter (str): choosen param

        Returns:
            the description tag as string
        """
        descr = self.connectors[system].description(parameter)

        return descr


    def print_params(self):
        print("System, Param")
        [print(k, self.connectors[k].params_list) for k in self.connectors.keys()]

    def days_with_data(self):
        """ 
        Returns:
            the number of files per day for each system
        """

        no_files = {}
        for system in self.connectors.keys():
            for which_path in self.connectors[system].filehandler.keys():
                no_files['system'] = self.connectors[system].files_per_day(which_path)
        return no_files


class LARDA_campaign:
    """ provides information about campaigns collected in LARDA"""
    def __init__(self, config_dir, campaign_file):

        logger.info('campaign file at LARDA_campaign ' + campaign_file)
        self.campaigns = toml.load(config_dir + campaign_file)
        self.campaign_list = list(self.campaigns.keys())
        self.config_dir = config_dir

    def get_campaign_list(self):
        """ list of all campaign names stored in csv-file """
        return self.campaign_list
        

    def assign_campaign(self, name):
        """dedicate object to a specific campaign"""
        
        
        self.info_dict = self.campaigns[name]

        logger.debug("info dict@assing campaign {}".format(self.info_dict))
        self.ALTITUDE=float(self.info_dict['altitude'])
        self.VALID_SYSTEMS = self.info_dict['systems']
        self.VALID_DATES = self.info_dict["duration"]
        self.COORDINATES = self.info_dict["coordinates"]
        self.CLOUDNET_STATIONNAME = self.info_dict["cloudnet_stationname"]
        self.CONFIGURATION_FILE = self.info_dict["param_config_file"]
        self.LOCATION = self.info_dict["location"]
        for i, val in enumerate(self.VALID_DATES):
            #begin = datetime.datetime.strptime(val[0], "%Y%m%d")
            if val[1] == 'today':
                end = datetime.datetime.utcnow().strftime("%Y%m%d")
            else:
                end = val[1]
            #    end = datetime.datetime.strptime(val[1], "%Y%m%d")
            self.VALID_DATES[i] = (val[0], end)


