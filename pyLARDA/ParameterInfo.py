#!/usr/bin/python3

import toml
import numpy as np
import pprint
import collections
import logging
import re
logger = logging.getLogger(__name__)


def deep_update(source, overrides):
    """Update a nested dictionary.
    Only additions, no removal modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def select_matching_template(syskey, templates, fname):
    """templates either match by exact name or regex in also_match

    Example:
        in a template
        ```
        [CLOUDNET]
          also_match = "CLOUDNET_.+"
          [CLOUDNET.generic]
          ...
        ```
        also matches ``CLOUDNET_LIMRAD``
    """
    if syskey in templates:
        return templates[syskey]
    else:
        logger.info('no direct hit in template, try also_match')
        # (?=a)b regex will never match
        matches = [
            k for k, v in templates.items() 
            if bool(re.findall(v.get('also_match', '(?=a)b'), syskey))]
        assert len(matches) == 1, \
            f'more than one matching pattern found in template, check also_match tag in {fname}'
        return templates[matches[0]]


def check_filter(filter, config):
    """ filter for a specific key in the config file, i.e. instr_name, instr_id, ...
    
    Args:
        filter: dict with name (instr_name) and matching value (MIRA_NMRA)
        config: dict loaded from the config file


    Returns:
        True or False


    TODO: maybe at one point we will need path level filtering
    
    """
    if filter[0] in config and config[filter[0]] == filter[1]:
        return True
    else:
        return False

def do_parameter_generic_inheritance(config, cinfo_hand_down={}):
    """inhert the parameter.generic information to the single variables

    Args:
        config (dict): config dict as read from toml
        cinfo_hand_down (dict, optional): campaign information to inherit

    Returns:
        config
    """

    # do the inheritance of system level parameters here
    for syskey, sysval in config.items():
        #print("system level keys ", 
        #      [e for e in sysval.keys()])
        defaults = {**cinfo_hand_down, **sysval['generic']}
        #defaults = {k: v for k, v in sysval['generic'].items()}
        #pprint.pprint(defaults)
        for pkey, pval in sysval["params"].items():
            #print("param", pkey)
            #logger.debug(
            #    "paraminfo "+ pprint.pformat({**defaults, **pval, **{'system': syskey, 'name': pkey}}))               
            config[syskey]["params"][pkey] = {
                **defaults, **pval, 
                **{'system': syskey, 'paramkey': pkey}}
            if 'meta' in defaults and 'meta' in pval:
                config[syskey]["params"][pkey]['meta'] = {
                    **defaults['meta'], **pval['meta']}

    return config


class ParameterInfo:
    """load the config file right here
    no need for prior allocation

    Args:
        config_path: location of the ``.toml`` config file
        config_file: name of the ``.toml`` config file
    """
    def __init__(self, config_path, config_file, cinfo_hand_down={}, root=None):
        logger.info("ParameterInfo: load config file {}".format(config_path, config_file))
        config = toml.load(config_path / config_file)
        self.config = {}
        
        for syskey, sysval in config.items():
            # get the template
            if 'template' in sysval:
                print(config_path / sysval['template'])
                # if local template is available use that one
                # otherwise fall back to default template
                if (config_path / sysval['template']).is_file():
                    templates = toml.load(config_path / sysval['template'])
                else:
                    #print('no local template available, check global')
                    templates = toml.load(root.parent / 'template_params' / sysval['template'])
                temp = select_matching_template(syskey, templates, sysval['template'])
                #print('template', temp.keys())
                sysval = deep_update(temp, sysval)
            self.config[syskey] = sysval

        self.config = do_parameter_generic_inheritance(self.config, cinfo_hand_down=cinfo_hand_down)
        

    def iterate_systems(self, keys=None, filter=None):
        """provide iterator for the systems structure
        
        Args:
            keys: just iterate over the given (i.e. valid) systems
        
        Yields:
            syskey, config
        """

        if filter:
            filtered_keys = [k for k,v in self.config.items() if check_filter(filter, v)]
        else:
            filtered_keys = self.config.keys()

        if keys == None:
            keys = filtered_keys
        else:
            keys = set(keys).intersection(filtered_keys)
        
        for syskey in keys:
            yield syskey, self.config[syskey]

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
