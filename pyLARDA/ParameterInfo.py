#!/usr/bin/python3

import toml
import numpy as np
import pprint
import logging
logger = logging.getLogger(__name__)

class ParameterInfo:
    def __init__(self,config_file, cinfo_hand_down={}):
        """

        new load the config file right here
        no need for prior allocation

        Args:
            config_file: location of the ``.toml`` config file
        """
        logger.info("ParameterInfo: load config file {}".format(config_file))
        self.config = toml.load(config_file)
        
        #logger.debug(self.config)
        
        # do the inheritance of system level parameters here
        for syskey, sysval in self.config.items():
            #print(syskey, sysval)
            #print("system level keys ", 
            #      [e for e in sysval.keys()])
            defaults = {**cinfo_hand_down, **sysval['generic']}
            #defaults = {k: v for k, v in sysval['generic'].items()}
            #pprint.pprint(defaults)
            for pkey, pval in sysval["params"].items():
                #print("param", pkey)
                #logger.debug(
                #    "paraminfo "+ pprint.pformat({**defaults, **pval, **{'system': syskey, 'name': pkey}}))               
                self.config[syskey]["params"][pkey] = {
                    **defaults, **pval, 
                    **{'system': syskey, 'paramkey': pkey}}
        

    def iterate_systems(self):
        """provide iterator for the systems structure"""
        for syskey in self.config.keys():
            yield syskey, self.config[syskey]

    def generate_cfg_list(self):
        """ list of all parameter configs stored in csv-file"""
        self.cfg_list = []
        for elem in self.csv_file:
            self.cfg_list.append(elem['PARAM_NAME'])
        return self.cfg_list
        

    def read_from_file(self,config_file,line):

        self.load_cfg_file(config_file)

        #info_dict = (item for item in self.csv_file if item['PARAM_NAME'] == param_name).next()
        info_dict = self.csv_file[line]
        #cfg_file=np.loadtxt(filename,"string")
        #lines=np.shape(cfg_file)[0]
	#print("ParameterInfo  infodict", info_dict)
        self.parameter_name = info_dict['PARAM_NAME']

        if "SYSTEM" in info_dict.keys():
            if not info_dict['SYSTEM'] == '':
                self.system_type = info_dict['SYSTEM']
        if "RANGE_MIN" in info_dict.keys():
            if not info_dict['RANGE_MIN'] == '': #check if string is empty
                self.value_range_min=float(info_dict['RANGE_MIN'])
        if "RANGE_MAX" in info_dict.keys():
            if not info_dict['RANGE_MAX'] == '':
                self.value_range_max=float(info_dict['RANGE_MAX'])
        if "COLORMAP" in info_dict.keys():
            if not info_dict['COLORMAP'] == '':
                self.colormap = info_dict['COLORMAP']
        if "DIMENSIONS" in info_dict.keys():
            if not info_dict['DIMENSIONS'] == '':
                self.dimensions = int(info_dict['DIMENSIONS'])
        if "VARIABLE_NAME" in info_dict.keys():
            if not info_dict['VARIABLE_NAME'] == '':
                self.variable_name = info_dict['VARIABLE_NAME']
        if "TIME_VARIABLE" in info_dict.keys():
            if not info_dict['TIME_VARIABLE'] == '':
                self.time_variable_name = info_dict['TIME_VARIABLE']
        if "TIME_DIMENSION" in info_dict.keys():
            if not info_dict['TIME_DIMENSION'] == '':
                self.time_dimension_name = info_dict['TIME_DIMENSION']
        if "RANGE_DIMENSION" in info_dict.keys():
            if not info_dict['RANGE_DIMENSION'] == '':
                self.range_dimension_name = info_dict['RANGE_DIMENSION']
        if "RANGE_RESOLUTION" in info_dict.keys():
            if not info_dict['RANGE_RESOLUTION'] == '':
                self.range_resolution = float(info_dict['RANGE_RESOLUTION'])
        if "ZERO_BIN_HEIGHT" in info_dict.keys():
            if not info_dict['ZERO_BIN_HEIGHT'] == '':
                self.zero_bin_height = float(info_dict['ZERO_BIN_HEIGHT'])
        if "ZERO_BIN" in info_dict.keys():
            if not info_dict['ZERO_BIN'] == '':
                self.zero_bin = int(info_dict['ZERO_BIN'])
        if "UNIT" in info_dict.keys():
            if not info_dict['UNIT'] == '':
                self.unit = info_dict['UNIT']
        if "DATA_DIR" in info_dict.keys():
            if not info_dict['DATA_DIR'] == '':
                self.data_dir = info_dict['DATA_DIR'].split(',')

        if "FILEMASK" in info_dict.keys():
            if not info_dict['FILEMASK'] == '':
                self.filemask = info_dict['FILEMASK']
        if "STORAGE_TYPE" in info_dict.keys():
            if not info_dict['STORAGE_TYPE'] == '':
                self.storage_type = info_dict['STORAGE_TYPE']
        if "DISPLAY_TYPE" in info_dict.keys():
            if not info_dict['DISPLAY_TYPE'] == '':
                self.display_type = info_dict['DISPLAY_TYPE']
        if "READ_LINE" in info_dict.keys():
            if not info_dict['READ_LINE'] == '':
                self.read_line = int(info_dict['READ_LINE'])
        if "CHANNEL" in info_dict.keys():
            if not info_dict['CHANNEL'] == '':
                self.channel = int(info_dict['CHANNEL'])
        if "FILE_EXTENSION" in info_dict.keys():
            if not info_dict['FILE_EXTENSION'] == '':
                self.file_extension = info_dict['FILE_EXTENSION']
        if "RANGE_INDEX" in info_dict.keys():
            if not info_dict['RANGE_INDEX'] == '':
                self.range_index = info_dict['RANGE_INDEX']
        #self.print_info()

    def name_and_unit(self):
                
        #if self.system_type=="Cloudnet":
            
        #    output = self.parameter_name
            
        #else :
        if self.unit == "" or self.unit == "[]":
            output = self.parameter_name
        else:
            if self.unit[0]=="[":
                output = self.parameter_name + " "+self.unit
            else:
                output = self.parameter_name + " ["+self.unit+"]"
        
        return output 
    
    def print_info(self):
                
        print("Parameter Name  : ",  self.parameter_name)
        print("System Type     : ",  self.system_type)
        print("Value Range Min : ",  self.value_range_min)
        print("Value Range Max : ",  self.value_range_max)
        print("COLORMAP        : ",  self.colormap)
        print("DIMENSIONS      : ",  self.dimensions)
        print("Variable  Name  : ",  self.variable_name)
        print("Time Var.       : ",  self.time_variable_name)
        print("Time Dim.       : ",  self.time_dimension_name)
        print("Range Dim.      : ",  self.range_dimension_name)
        print("Range Resolution: ",  self.range_resolution)
        print("Zero Bin        : ",  self.zero_bin)
        print("Zero Bin Height : ",  self.zero_bin_height)
        print("Unit            : ",  self.unit)
        print("Data Directory  : ",  self.data_dir)
        print("Filemask        : ",  self.filemask)
        print("Storage Type    : ",  self.storage_type)
        print("Display Type    : ",  self.display_type)
        print("Read Line       : ",  self.read_line)
        print("File Extension  : ",  self.file_extension)
