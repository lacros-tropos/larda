#!/usr/bin/python3


import pyLARDA.Connector as Connector
import pyLARDA.ParameterInfo as ParameterInfo
import pyLARDA.IO
import pyLARDA.VIS_Html2

import datetime, os, calendar, copy, time
import numpy as np
import csv
import logging
import toml


class LARDA :
    """
    built the connector list for the specified campaign (only valid systems are considered)
        the connectors are instances of the Connector.Connector Class


    NEW: one connector per system
    then the params are parts of this system
    """
    def __init__(self, camp_name, campaign=False, build_lists=True):

        logger = logging.getLogger("pyLARDA")
        logger.info(' pyLARDA module')

        self.connectors={}

        if not campaign:
            self.camp = LARDA_campaign("/home/larda/larda-cfg/", 'campaigns.toml')
        else:
            self.camp = campaign
        logger.info("campaign list " + ' '.join(self.camp.get_campaign_list()))

        print("camp_name set ", camp_name)

        self.camp.assign_campaign(camp_name)
        config_file=self.camp.CONFIGURATION_FILE
        cinfo_hand_down = {'coordinates': self.camp.COORDINATES,
                           'altitude': self.camp.ALTITUDE,
                           'location': self.camp.LOCATION}
        print("config file ", config_file)
        
        paraminformation = ParameterInfo.ParameterInfo(
            self.camp.config_dir + config_file, 
            cinfo_hand_down=cinfo_hand_down)
        starttime = time.time()
        print("camp.VALID_SYSTEMS ", self.camp.VALID_SYSTEMS)
        print("LARDA :", config_file)
        
        #if camp_name == 'LACROS_at_Leipzig':
        #    build_lists = False
        # build the filelists or load them from json
        for system, systeminfo in paraminformation.iterate_systems():
            print('current parameter ', system, systeminfo)

            conn = Connector.Connector(system, systeminfo,
                                       self.camp.VALID_DATES)
            
            if build_lists:
                conn.build_filehandler()
                conn.save_filehandler(self.camp.info_dict['connectordump'])

            else:
                #load lists
                conn.load_filehandler()
            
            if system in self.camp.VALID_SYSTEMS:
                self.connectors[system] = conn
            else:
                print(system, "not in valid systems")

        print('Time for generating connectors ', time.time() - starttime, '[s]')
        #print "Availability array: ", self.array_avail(2014, 2)
        print(self.connectors.keys())
        print("Parameters in stock: ",[(k, self.connectors[k].params_list) for k in self.connectors.keys()])
        
    

    def read(self,system,parameter,time_interval,*further_slices):
        """
        
        Args:
            system (str): identifier for the system
            parameter (str): choosen param
            time_interval: ``[dt, dt]`` time interval
            *further_slices: range, vel,.. ``[0, max]``

        Returns:
            the dictionary with data
        """
        print(self.connectors[system])
        
        data = self.connectors[system].collect(parameter, time_interval, *further_slices) 

        return data

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

        logger = logging.getLogger("pyLARDA")
        self.campaigns = toml.load(config_dir + campaign_file)
        logger.debug('campaign file at LARDA_campaign ' + campaign_file)
        self.campaign_list = list(self.campaigns.keys())
        self.config_dir = config_dir

    def get_campaign_list(self):
        """ list of all campaign names stored in csv-file """
        return self.campaign_list
        

    def assign_campaign(self, name):
        """dedicate object to a specific campaign"""
        
        
        self.info_dict = self.campaigns[name]

        print(self.info_dict)
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


