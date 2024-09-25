
#!/usr/bin/python3

"""

"""

import numpy as np
import pyLARDA.helpers as h
from pyLARDA.NcReader import get_time_slicer, get_var_attr_from_nc
#from typing import List
import logging

logger = logging.getLogger(__name__)

def get_meta_from_bD(bD, meta_spec, varname):
    """get some meta data into the data_container

    specified within the paraminfo meta.name tags
    - gattr.name: global attribute with name
    - var.name: additional varable (ideally single value)

    Args:
        ncD: netCDF file object
        meta_spec: dict with all meta. definition
        varname: name of the variable to load (for vattr)

    Returns:
        dict with meta data
    """

    meta = {}
    for k, v in meta_spec.items():
        where, name = v.split('.')
        if where == 'gattr':
            meta[k] = [bD[name]]
        else:
            raise ValueError(f'meta string {v} for {k} not specified')
    return meta


def rpgfmcw_binary(paraminfo):
    """build a function for reading in time height data
    """

    def retfunc(f, time_interval, *further_intervals):
        """reading the rpg94 data with rpgypy and convert into the larda-data-format
        """
        from rpgpy import read_rpg, spectra2moments

        logger.debug(f"filename at rpgpy binary {f}")
        header, data = read_rpg(str(f))

        logger.debug(f'Header: {header.keys()}')
        logger.debug(f'Data  : {data.keys()}')

        # bD binary Data (in resemblance to ncD)
        if paraminfo['ncreader'] == 'spec_rpg94binary' and paraminfo['variable_name'] in ['Ze', 'MeanVel', 'SpecWidth', 'Skewn', 'Kurt']:
            moments = spectra2moments(data, header)
            bD = {**header, **data, **moments}
        else:
            bD = {**header, **data}

        if paraminfo['ncreader'] in ['timeheight_rpg94binary', 'spec_rpg94binary']:
            try:
                range_interval = further_intervals[0]
            except:
                range_interval = []
            ranges = bD[paraminfo['range_variable']][:].astype(np.float64)
        
        times = bD[paraminfo['time_variable']][:].astype(np.float64)
        if 'time_millisec_variable' in paraminfo.keys() and \
                paraminfo['time_millisec_variable'] in bD:
            subsec = bD[paraminfo['time_millisec_variable']][:] / 1.0e3
            times += subsec
        if 'time_microsec_variable' in paraminfo.keys() and \
                paraminfo['time_microsec_variable'] in bD:
            subsec = bD[paraminfo['time_microsec_variable']][:] / 1.0e6
            times += subsec
        timeconverter, _ = h.get_converter_array(
            paraminfo['time_conversion'])
        ts = timeconverter(times)
        
        # get the time slicer from time_interval
        slicer = get_time_slicer(ts, f, time_interval)
        if slicer == None:
            return None
        
        varconverter, _ = h.get_converter_array(
            paraminfo['var_conversion'])

        if paraminfo['ncreader'] in ['timeheight_rpg94binary', 'spec_rpg94binary']:
            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])
            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))
        
        var = bD[paraminfo['variable_name']]
        print(f"var shape {paraminfo['variable_name']} {var.shape}")
        data = {}
        if paraminfo['ncreader'] in ['timeheight_rpg94binary', 'spec_rpg94binary'] and len(var.shape) == 2:
            data['dimlabel'] = ['time', 'range']
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
        elif paraminfo['ncreader'] in ['time_rpg94binary']:
            data['dimlabel'] = ['time']
        elif paraminfo['ncreader'] == 'spec_rpg94binary' and len(var.shape) == 3:
            data['dimlabel'] = ['time', 'range', 'vel']
            # TODO think of a better solution for different velocity vectors
            #  in different chirps
            data['vel'] = bD[paraminfo['vel_variable']][:].astype(np.float64)
        data["filename"] = f
        data["paraminfo"] = paraminfo
        data['ts'] = ts[tuple(slicer)[0]]

        data['system'] = paraminfo['system']
        data['name'] = paraminfo['paramkey']
        data['colormap'] = paraminfo['colormap']

        if 'meta' in paraminfo:
            data['meta'] = get_meta_from_bD(bD, paraminfo['meta'], paraminfo['variable_name'])
        if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
            data['plot_varconverter'] = paraminfo['plot_varconverter']
        else:
            data['plot_varconverter'] = ''

        if paraminfo['ncreader'] in ['timeheight_rpg94binary', 'spec_rpg94binary']:
            if isinstance(times, np.ma.MaskedArray):
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]].data)
            else:
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ranges)

        data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                paraminfo, var)
        data['var_lims'] = [float(e) for e in \
                            get_var_attr_from_nc("identifier_var_lims",
                                                 paraminfo, var)]

        if "identifier_var_def" in paraminfo.keys() and not "var_def" in paraminfo.keys():
            data['var_definition'] = h.guess_str_to_dict(
                var.getncattr(paraminfo['identifier_var_def']))
        elif "var_def" in paraminfo.keys():
            data['var_definition'] =  paraminfo['var_def']

        data['var'] = varconverter(var[:])[tuple(slicer)]        

        # no getncattr available for binary data
        #if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
        #    fill_value = var.getncattr(paraminfo['identifier_fill_value'])
        #    mask = np.isclose(data['var'].data, fill_value)
        if "fill_value" in paraminfo.keys():
            fill_value = paraminfo['fill_value']
            mask = np.isclose(data['var'].data, fill_value)
        else:
            mask = ~np.isfinite(data['var'].data)

        assert not isinstance(mask, np.ma.MaskedArray), \
           "mask array shall not be np.ma.MaskedArray, but of plain booltype"
        data['mask'] = mask

        if isinstance(data['var'], np.ma.MaskedArray):
            data['var'] = data['var'].data
        assert not isinstance(data['var'], np.ma.MaskedArray), \
           "var array shall not be np.ma.MaskedArray, but of plain booltype"

        return data

    return retfunc



def read_hatpro(f):
    """"based on get_datasets in plotting_hatpro-raw by 
    
    
    ## set dtypes of the different hatpro_raw binary files
    """

    pattern = f.suffix[1:]
    print(f, pattern)
    
    if pattern == 'AbsH' or pattern == 'RelH':
        file_pattern = 'HPC'
    else:
        file_pattern = pattern
    
    # do not loop trough the files
    ## House Keeping Data (HKD)
    if pattern=="HKD":
        
        ## read bits of first byte of 'select'
        select_Byte = np.fromfile(f, dtype = "uint8",offset=12,count=1)
        select_bits = np.unpackbits(select_Byte)[::-1]
        print('HKD select_bits', select_bits)
        ## set flags depending on select_bits
        # probably the manual changes from 0-based (MET) bit 
        # to 1-based (HKD) indices
        long_lat_flag = 1 if select_bits[0] == 1 else 0 # Bit 1: When this bit is set to ‘1’, the GPS-position (longitude, latitude) is recorded  
        temp_flag = 4 if select_bits[1] == 1 else 3 # Bit 2: When this bit is set to ‘1’, the temperature data is recorded
        stab_flag = 2 if select_bits[2] == 1 else 0 # Bit 3: When this bit is set to ‘1’, the receiver stability data is recorded
        mem_flag = 1 if select_bits[3] == 1 else 0 # Bit 4: When this bit is set to ‘1’, the remaining flash memory is recorded
        qual_flag = 1 if select_bits[4] == 1 else 0 # Bit 5: When this bit is set to ‘1’, quality flags are recorded
        stat_flag = 1 if select_bits[5] == 1 else 0 # Bit 6: When this bit is set to ‘1’, status flags are recorded
        
        print(long_lat_flag,temp_flag,stab_flag,mem_flag,qual_flag,stat_flag)
        
        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                            ('n', np.uint32),
                            ('timref', np.int32),
                            ('select', np.int32)
                            ])
        ## data-section
        dt_data = np.dtype([('mactime',np.int32),
                            ('alarm', 'u1'),
                            ('long', np.float32,long_lat_flag),
                            ('lat', np.float32,long_lat_flag),
                            ('temp', np.float32,temp_flag),
                            ('stab', np.float32,stab_flag),
                            ('mem', np.int32,mem_flag),
                            ('qual', np.int32,qual_flag),
                            ('status', np.int32,stat_flag)
                            ])

        
    ## Meteorological Data (MET)
    elif pattern=="MET":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)

        if code_[0]['code'] == 599658943:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min_p', np.float32),
                                ('max_p', np.float32),
                                ('min_T', np.float32),
                                ('max_T', np.float32),
                                ('min_H', np.float32),
                                ('max_H', np.float32),
                                ('timref', np.int32),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var_p', np.float32),
                                ('var_T', np.float32),
                                ('var_H', np.float32)
                                ])

        if code_[0]['code'] == 599658944:
            ## read bits of byte 'addsensors'
            addsensors_Byte = np.fromfile(f, dtype = "u1",offset=8,count=1)
            addsensors_bits = np.unpackbits(addsensors_Byte)[::-1]
            ## Additional sensors bit field:
            ##      Bit0 (LSB): wind speed (km/h)
            ##      Bit1: wind direction [°]
            ##      Bit2: Rain Rate.
            ##  If corresponding bit is 1, the additional sensor exists, otherwise it does not
            
            ## set flags depending on addsensors_bits
            add_1_flag = 1 if addsensors_bits[0] == 1 else 0
            add_2_flag = 1 if addsensors_bits[1] == 1 else 0
            add_3_flag = 1 if addsensors_bits[2] == 1 else 0
            #print('add sensors ', add_1_flag, add_2_flag, add_3_flag)
            
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('addsensors', 'u1'),
                                ('min_p', np.float32),
                                ('max_p', np.float32),
                                ('min_T', np.float32),
                                ('max_T', np.float32),
                                ('min_H', np.float32),
                                ('max_H', np.float32),
                                ('min_add_1', np.float32,add_1_flag),
                                ('max_add_1', np.float32,add_1_flag),
                                ('min_add_2', np.float32,add_2_flag),
                                ('max_add_2', np.float32,add_2_flag),
                                ('min_add_3', np.float32,add_3_flag),
                                ('max_add_3', np.float32,add_3_flag),
                                ('timref', np.int32),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var_p', np.float32),
                                ('var_T', np.float32),
                                ('var_H', np.float32),
                                ('var_add_1', np.float32,add_1_flag),
                                ('var_add_2', np.float32,add_2_flag),
                                ('var_add_3', np.float32,add_3_flag),
                                ])


    ## Atmospheric Attenuation
    elif pattern=="ATN":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        # get number of frequencies
        frnr = np.dtype([('frnr', np.uint32)])
        frnr_= np.fromfile(f, dtype=frnr,offset=16,count=1)
        freqnr=frnr_[0]['frnr']

        if code_[0]['code'] == 7757564:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('timref', np.int32),
                                ('retr', np.int32),
                                ('freqnr',np.uint32),
                                ('frequs',np.float32,freqnr),
                                ('min', np.float32,freqnr),
                                ('max', np.float32,freqnr),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.float32)
                                ])

        if code_[0]['code'] == 7757000:
            # get number of frequencies
            frnr = np.dtype([('frnr', np.uint32)])
            irt_frnr= np.fromfile(f, dtype=frnr,offset=16,count=1)
            freqnr=irt_frnr[0]['frnr']
            
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('timref', np.int32),
                                ('retr', np.int32),
                                ('freqnr',np.uint32),
                                ('frequs',np.float32,freqnr),
                                ('min', np.float32,freqnr),
                                ('max', np.float32,freqnr),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.int32)
                                ])


    ## Integrated Water Vapor
    elif pattern=="IWV":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        if code_[0]['code'] == 594811068:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min', np.float32),
                                ('max', np.float32),
                                ('timref', np.int32),
                                ('retr', np.int32)
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32),
                                ('var_a', np.float32)
                                ])
        if code_[0]['code'] == 594811000:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min', np.float32),
                                ('max', np.float32),
                                ('timref', np.int32),
                                ('retr', np.int32)
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32),
                                ('var_a', np.int32)
                                ])
        

    ## Liquid Water Path
    elif pattern=="LWP":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        if code_[0]['code'] == 934501978:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min', np.float32),
                                ('max', np.float32),
                                ('timref', np.int32),
                                ('retr', np.int32)
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32),
                                ('var_a', np.float32)
                                ])
        if code_[0]['code'] == 934501000:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min', np.float32),
                                ('max', np.float32),
                                ('timref', np.int32),
                                ('retr', np.int32)
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32),
                                ('var_a', np.int32)
                                ])


    ## Cloud Base Height (CBH)
    elif pattern=="CBH":
        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                            ('n', np.uint32),
                            ('min', np.float32),
                            ('max', np.float32),
                            ('timref', np.int32)
                            ])
        ## data-section
        dt_data = np.dtype([('mactime',np.int32),
                            ('rf', 'u1'),
                            ('var', np.float32)
                            ])

        
    ## Boundary Layer Height (BLH)
    elif pattern=="BLH":
        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                            ('n', np.uint32),
                            ('min', np.float32),
                            ('max', np.float32),
                            ('timref', np.int32)
                            ])
        ## data-section
        dt_data = np.dtype([('mactime',np.int32),
                            ('rf', 'u1'),
                            ('var', np.float32)
                            ])


    ## Infrared Temperature
    elif pattern=="IRT":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        if code_[0]['code'] == 671112495:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('min', np.float32),
                                ('max', np.float32),
                                ('timref', np.int32)
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32)
                                ])
                    
        if code_[0]['code'] == 671112496:
            # get number of frequencies
            frnr = np.dtype([('frnr', np.uint32)])
            irt_frnr= np.fromfile(f, dtype=frnr,offset=20,count=1)
            freqnr=irt_frnr[0]['frnr']
            
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                            ('n', np.uint32),
                            ('min', np.float32),
                            ('max', np.float32),
                            ('timref', np.int32),
                            ('freqnr',np.uint32),
                            ('frequs',np.float32,freqnr)
                            ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.float32)
                                ])

        if code_[0]['code'] == 671112000:
            # get number of frequencies
            frnr = np.dtype([('frnr', np.uint32)])
            irt_frnr= np.fromfile(f, dtype=frnr,offset=20,count=1)
            freqnr=irt_frnr[0]['frnr']
            
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                            ('n', np.uint32),
                            ('min', np.float32),
                            ('max', np.float32),
                            ('timref', np.int32),
                            ('freqnr',np.uint32),
                            ('frequs',np.float32,freqnr)
                            ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.int32)
                                ])


    ## Brightness Temperature
    elif pattern=="BRT":
        ## check code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        # get number of frequencies
        frnr = np.dtype([('frnr', np.uint32)])
        frnr_= np.fromfile(f, dtype=frnr,offset=12,count=1)
        freqnr=frnr_[0]['frnr']

        if code_[0]['code'] == 666666 or code_[0]['code'] == 666667:
            ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('timref', np.int32),
                                ('freqnr',np.uint32),
                                ('frequs',np.float32,freqnr),
                                ('min', np.float32,freqnr),
                                ('max', np.float32,freqnr),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.float32)
                                ])

        if code_[0]['code'] == 666000 or code_[0]['code'] == 667000:
            # get number of frequencies
            frnr = np.dtype([('frnr', np.uint32)])
            irt_frnr= np.fromfile(f, dtype=frnr,offset=12,count=1)
            freqnr=irt_frnr[0]['frnr']
            print(freqnr)
            
            ## header-section
             ## header-section
            dt_head = np.dtype([('code', np.uint32),
                                ('n', np.uint32),
                                ('timref', np.int32),
                                ('freqnr',np.uint32),
                                ('frequs',np.float32,freqnr),
                                ('min', np.float32,freqnr),
                                ('max', np.float32,freqnr),
                                ])
            ## data-section
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,freqnr),
                                ('var_a', np.int32)
                                ])


    ## Full Tropospheric temperature profile charts (TPC)
    elif pattern=="TPC":
        # get number of altitude layers
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)

        altl = np.dtype([('altl', np.uint32)])
        altl_= np.fromfile(f, dtype=altl,offset=24,count=1)
        alt_layers=altl_[0]['altl']
        
        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                        ('n', np.uint32),
                        ('min', np.float32),
                        ('max', np.float32),
                        ('timref', np.int32),
                        ('retr', np.int32),
                        ('altanz', np.int32),
                        ('alts',np.int32,alt_layers)
                        ])
        ## data-section
        if code_[0]['code'] == 780798065:
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,alt_layers)
                                ])
        elif code_[0]['code'] == 780798066:
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,alt_layers),
                                ('elaz', np.int32),
                                ('ra', np.float32),
                                ('dec', np.float32),
                                ])
        else:
            raise ValueError('File code not implemented')

        
    ## Temperature profile of boundary layer (TPB)
    elif pattern=="TPB":
        # get number of altitude layers
        altl = np.dtype([('altl', np.uint32)])
        altl_= np.fromfile(f, dtype=altl,offset=24,count=1)
        alt_layers=altl_[0]['altl']
        
        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                        ('n', np.uint32),
                        ('min', np.float32),
                        ('max', np.float32),
                        ('timref', np.int32),
                        ('retr', np.int32),
                        ('altanz', np.int32),
                        ('alts',np.int32,alt_layers)
                        ])
        ## data-section
        dt_data = np.dtype([('mactime',np.int32),
                            ('rf', 'u1'),
                            ('var', np.float32,alt_layers)
                            ])
    

    ## AbsHumidity profile chart from HPC-file (AbsH)
    elif pattern=="HPC":
        
        ## check HPC-code
        code = np.dtype([('code', np.uint32)])
        code_ = np.fromfile(f, dtype=code,count=1)
        
        # get number of altitude layers
        altl = np.dtype([('altl', np.uint32)])
        altl_= np.fromfile(f, dtype=altl,offset=24,count=1)
        alt_layers=altl_[0]['altl']
        
        # check number of samples
        samples = np.dtype([('samples', np.uint32)])
        samples_= np.fromfile(f, dtype=samples,offset=4,count=1)
        samples = samples_[0]['samples']

        ## header-section
        dt_head = np.dtype([('code', np.uint32),
                        ('n', np.uint32),
                        ('min', np.float32),
                        ('max', np.float32),
                        ('timref', np.int32),
                        ('retr', np.int32),
                        ('altanz', np.int32),
                        ('alts',np.int32,alt_layers)
                        ])
        
        ## data-section
        if code_[0]['code'] in [117343672, 117343673]: 
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,alt_layers)
                                ])
        elif code_[0]['code'] in [117343674, 117343675]: 
            dt_data = np.dtype([('mactime',np.int32),
                                ('rf', 'u1'),
                                ('var', np.float32,alt_layers),
                                ('elaz', np.int32),
                                ('ra', np.float32),
                                ('dec', np.float32),
                                ])
        else:
            raise ValueError('File code not implemented')
            
        head = np.fromfile(f, dtype=dt_head, count=1)
        data = np.fromfile(f, dtype=dt_data, offset=dt_head.itemsize, count=samples)

    else:
        raise ValueError(f'pattern not found {pattern}')
    
    ###
    ## write to dict
    ###

    if not pattern=='HPC':
        head = np.fromfile(f, dtype=dt_head, count=1)
        data = np.fromfile(f, dtype=dt_data, offset=dt_head.itemsize, count=-1)
    

    # returns a Structured Array
    print(dt_head, head.shape, head.dtype)
    print(dt_data, data.shape, data.dtype)
    print('mactime ', data['mactime'])
    return head, data



def hatpro_binary(paraminfo):
    """build a function for reading in time height data
    """

    def retfunc(f, time_interval, *further_intervals):
        """reading the hatpro data and convert into the larda-data-format
        """

        logger.debug(f"filename {f} {type(f)}")
        header, data = read_hatpro(f)

        logger.debug(f'Header: {header.dtype}')
        logger.debug(f'Data  : {data.dtype}')

        # bD binary Data (in resemblance to ncD)
        bD = {}
        for col in header.dtype.names:
            bD[col] = header[col]
        for col in data.dtype.names:
            bD[col] = data[col]
        logger.debug(f'binaryData.keys(): {bD.keys()}')

        if paraminfo['ncreader'] in ['timeheight_hatprobinary']:
            try:
                range_interval = further_intervals[0]
            except:
                range_interval = []
            ranges = bD[paraminfo['range_variable']][:].astype(np.float64).ravel()
        
        times = bD[paraminfo['time_variable']][:].astype(np.float64)
        if 'time_millisec_variable' in paraminfo.keys() and \
                paraminfo['time_millisec_variable'] in bD:
            subsec = bD[paraminfo['time_millisec_variable']][:] / 1.0e3
            times += subsec
        if 'time_microsec_variable' in paraminfo.keys() and \
                paraminfo['time_microsec_variable'] in bD:
            subsec = bD[paraminfo['time_microsec_variable']][:] / 1.0e6
            times += subsec
        timeconverter, _ = h.get_converter_array(
            paraminfo['time_conversion'])
        ts = timeconverter(times)
        
        # get the time slicer from time_interval
        slicer = get_time_slicer(ts, f, time_interval)
        if slicer == None:
            return None
        
        varconverter, _ = h.get_converter_array(
            paraminfo['var_conversion'])

        if paraminfo['ncreader'] in ['timeheight_hatprobinary']:
            rangeconverter, _ = h.get_converter_array(
                paraminfo['range_conversion'])
            print(ranges.shape, rangeconverter(ranges[:]), range_interval)
            ir_b = h.argnearest(rangeconverter(ranges[:]), range_interval[0])
            if len(range_interval) == 2:
                if not range_interval[1] == 'max':
                    ir_e = h.argnearest(rangeconverter(ranges[:]), range_interval[1])
                    ir_e = ir_e + 1 if not ir_e == ranges.shape[0] - 1 else None
                else:
                    ir_e = None
                slicer.append(slice(ir_b, ir_e))
            else:
                slicer.append(slice(ir_b, ir_b + 1))
        
        var = bD[paraminfo['variable_name']]
        print(f"var shape {paraminfo['variable_name']} {var.shape}")
        data = {}
        if paraminfo['ncreader'] in ['timeheight_hatprobinary']:
            data['dimlabel'] = ['time', 'range']
            data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
        elif len(var.shape) > 1:
            data['dimlabel'] = ['time', 'aux']
        else:
            data['dimlabel'] = ['time']
        data["filename"] = f
        data["paraminfo"] = paraminfo
        data['ts'] = ts[tuple(slicer)[0]]

        data['system'] = paraminfo['system']
        data['name'] = paraminfo['paramkey']
        data['colormap'] = paraminfo['colormap']

        if 'meta' in paraminfo:
            data['meta'] = get_meta_from_bD(bD, paraminfo['meta'], paraminfo['variable_name'])
        if 'plot_varconverter' in paraminfo and paraminfo['plot_varconverter'] != 'none':
            data['plot_varconverter'] = paraminfo['plot_varconverter']
        else:
            data['plot_varconverter'] = ''

        if paraminfo['ncreader'] in ['timeheight_hatprobinary']:
            if isinstance(times, np.ma.MaskedArray):
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]].data)
            else:
                data['rg'] = rangeconverter(ranges[tuple(slicer)[1]])
            data['rg_unit'] = get_var_attr_from_nc("identifier_rg_unit",
                                                   paraminfo, ranges)

        data['var_unit'] = get_var_attr_from_nc("identifier_var_unit",
                                                paraminfo, var)
        data['var_lims'] = [float(e) for e in \
                            get_var_attr_from_nc("identifier_var_lims",
                                                 paraminfo, var)]

        if "identifier_var_def" in paraminfo.keys() and not "var_def" in paraminfo.keys():
            data['var_definition'] = h.guess_str_to_dict(
                var.getncattr(paraminfo['identifier_var_def']))
        elif "var_def" in paraminfo.keys():
            data['var_definition'] =  paraminfo['var_def']

        data['var'] = varconverter(var[:])[tuple(slicer)]        

        # no getncattr available for binary data
        #if "identifier_fill_value" in paraminfo.keys() and not "fill_value" in paraminfo.keys():
        #    fill_value = var.getncattr(paraminfo['identifier_fill_value'])
        #    mask = np.isclose(data['var'].data, fill_value)
        if "fill_value" in paraminfo.keys():
            fill_value = paraminfo['fill_value']
            mask = np.isclose(data['var'].data, fill_value)
        else:
            mask = ~np.isfinite(data['var'].data)

        assert not isinstance(mask, np.ma.MaskedArray), \
           "mask array shall not be np.ma.MaskedArray, but of plain booltype"
        data['mask'] = mask

        if isinstance(data['var'], np.ma.MaskedArray):
            data['var'] = data['var'].data
        assert not isinstance(data['var'], np.ma.MaskedArray), \
           "var array shall not be np.ma.MaskedArray, but of plain booltype"

        return data

    return retfunc