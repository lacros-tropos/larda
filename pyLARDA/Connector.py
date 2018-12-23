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


class Connector:    
    """ """
    def __init__(self, system, system_info, valid_dates):
        self.system=system
        self.system_info=system_info
        self.valid_dates=valid_dates
        self.params_list = system_info["params"].keys()
        print("params in this connector", self.params_list)
        print('connector.system_info', system_info)


    def __str__(self):
        s = "connector for system {} \ncontains parameters: ".format(self.system)
        s += " ".join(self.params_list)
        return s


    def build_filehandler(self):
        """ """
        print("this is filelist")
        pprint.pprint(self.system_info)

        pathdict = self.system_info['path']

        filehandler = {}
        for key, pathinfo in pathdict.items():
            all_files = []
            for root, dirs, files in os.walk(pathinfo['base_dir']):
                print(root, dirs, len(files), files[:5], files[-5:] )
                current_regex = pathinfo['matching_subdirs']
                abs_filepaths = [root +'/'+ f for f in files if re.search(current_regex, root +'/'+ f)]
    
                all_files += abs_filepaths
                #files = [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]
    
            # remove basedir (not sure if that is a good idea)
            all_files = [p.replace(pathinfo['base_dir'], "./") for p in all_files]
            print('filelist ', len(all_files), all_files[:10])
    
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
            #if "IWC" in self.system_info['params']:
            #    exit()
            # maybe dict is not the solution
            #singlehandler = collections.defaultdict(
            #    lambda: {'files': [], 'time_ranges': []})
            #for f in sorted(all_files):
            #    re_date = re.search(pathinfo["date_in_filename"], f)
            #    #print(f, re_date.groupdict())
            #    dt = convert_regex_date_to_dt(re_date.groupdict())
            #    singlehandler[dt.strftime('%Y%m%d')]['files'] += [f]
            #    singlehandler[dt.strftime('%Y%m%d')]['time_ranges'] += [[dt.strftime('%Y%m%d-%H%M'),]]
            
                #NcReader.peek_file(pathdict[key]['base_dir']+f, pathinfo['time_variable'])
                
            filehandler[key] = singlehandler
        #pprint.pprint(filehandler)
        self.filehandler = filehandler 


    def save_filehandler(self, path):
        savename = 'connector_{}.json'.format(self.system)
        pretty = {'indent': 2, 'sort_keys':True}
        #pretty = {}

        with open(path+savename, 'w') as outfile:
                json.dump(self.filehandler, outfile, **pretty)
                print('saved connector to ', path+savename)


    def load_filehandler(self, path):
        filename = path + "connector_{}.json".format(self.system)
        with open(path+savename) as json_data:
                self.filehandler = json.load(json_data)


    def collect(self, param, time_interval, *further_intervals):
        """collect the data from a parameter for the given intervals

        Args:
            time_interval: list of begin and end datetime
            *further_intervals: range, velocity, ...
        """
        
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


                
    def collect_old(self, begin_time, end_time, begin_height=0, end_height=0):
        """ """
        output=0
        
        if (self.param_info.system_type=="GDAS" or self.param_info.system_type=="WRF"):
            output=self.collectMETEO(begin_time, end_time, begin_height, end_height)
            
        elif (self.param_info.dimensions==1):
            output=self.collect1D(begin_time, end_time)
        
        elif (self.param_info.dimensions==2):
            output=self.collect2D(begin_time, end_time, begin_height, end_height)
        elif (self.param_info.dimensions==3):
            output=self.collectSPEC(begin_time, end_time, begin_height, end_height)
            
        output.restore()
        output.convert()
        
        return output
        
        
    def collect1D(self,begin_time, end_time):
        """ """
        matching_files = self.get_matching_files(begin_time, end_time)
        data_buffer=DataBuffer.DataBuffer()

        n=0
        for i in range(len(matching_files)):
            time_boundaries=self.time_index_boundaries(matching_files[i], begin_time, end_time)
            if time_boundaries[0]==0 and time_boundaries[1]==0:
                continue
            data_buffer.times=np.concatenate((data_buffer.times,self.read_time(matching_files[i], time_boundaries[0], time_boundaries[1])))
            data_buffer.data=np.concatenate((data_buffer.data,self.read_1D_data(matching_files[i], time_boundaries[0], time_boundaries[1])))
            n+=1
            
        data_buffer.selected_begin_time=begin_time
        data_buffer.selected_end_time=end_time
        data_buffer.begin_time=data_buffer.times[0]
        data_buffer.end_time=data_buffer.times[len(data_buffer.times)-1]
        
        data_buffer.param_info=self.param_info
        
        data_buffer.build_time_index()
        #data_buffer.build_height_index()
        return data_buffer
            
    def collect2D(self,begin_time, end_time, bottom_height, top_height):
        """ """
        matching_files = self.get_matching_files(begin_time, end_time)
        data_buffer=DataBuffer.DataBuffer()

        if self.param_info.system_type=="ICON":
            matlab=True
        else:
            matlab=False

        n=0
        for i in range(len(matching_files)):
            time_boundaries=self.time_index_boundaries(matching_files[i], begin_time, end_time)
            if time_boundaries[0]==0 and time_boundaries[1]==0:
                continue
            
            height_boundaries=self.height_index_boundaries(matching_files[i], bottom_height, top_height)
            data_buffer.times=np.concatenate((data_buffer.times,self.read_time(matching_files[i], time_boundaries[0], time_boundaries[1])))
            if n==0:
                data_buffer.data=np.empty((0,height_boundaries[1]-height_boundaries[0]))
            data_buffer.data=np.concatenate((data_buffer.data,self.read_2D_data(matching_files[i], time_boundaries[0], time_boundaries[1], height_boundaries[0], height_boundaries[1], matlab)))
            n+=1
        #print("data_buffer.times ", data_buffer.times)
        data_buffer.selected_begin_time=begin_time
        data_buffer.selected_end_time=end_time
        data_buffer.begin_time=data_buffer.times[0]
        data_buffer.end_time=data_buffer.times[len(data_buffer.times)-1]
        data_buffer.bottom_index=height_boundaries[0]
        data_buffer.top_index=height_boundaries[1]
        data_buffer.param_info=self.param_info
        #print("databuffer ", begin_time, end_time)
        #print("databuffer ", data_buffer.times[0], data_buffer.times[len(data_buffer.times)-1], len(data_buffer.times))

        if data_buffer.param_info.parameter_name=="WIPRO_VEL" or data_buffer.param_info.parameter_name=="WIPRO_VEL_IOP":
            data_buffer.data=data_buffer.data*(-1.0)

        data_buffer.build_time_index()
        if self.param_info.range_index != "":
            height_map = NcReader.read_1D(matching_files[0], self.param_info.range_index)
            
            #if data_buffer.param_info.system_type=="ICON":
                #height_map=np.flipud(height_map)
                #data_buffer.data=np.fliplr(data_buffer.data)
                #data_buffer.data=np.rot90(data_buffer.data)
                #data_buffer.data=np.rollaxis(data_buffer.data,1,0)
            print("Has range index")
            data_buffer.build_height_index(height_map)
            #print(height_map)
            data_buffer.height_map = height_map
        else:
            print("Has no range index")
            data_buffer.build_height_index()
        return data_buffer
    

    def collectSPEC(self,begin_time, end_time, bottom_height, top_height):
        """ """
        matching_files = self.get_matching_files(begin_time, end_time)
        # init empy data buffer
        data_buffer=DataBuffer.DataBufferSpec()

        n=0
        print('starting collectSPEC', len(matching_files))
        #assert len(matching_files) == 1, 'multiple files not supported yet'
 
        for i, file in enumerate(matching_files):
            datadict = Spec.load_mira_spec(file, self.param_info, begin_time, end_time,
                                           bottom_height, top_height)
            #data_buffer.times=np.concatenate((data_buffer.times, datadict['time']))
            #data_buffer.data=np.concatenate((data_buffer.data,datadict['value']))
            if i == 0:
                data_buffer.times = datadict['time']
                data_buffer.range = datadict['range']
                data_buffer.velocity = datadict['velocity']
                data_buffer.data = datadict['value']
            else:
                data_buffer.times = np.append(data_buffer.times, datadict['time'], axis=0)
                data_buffer.data = np.append(data_buffer.data, datadict['value'], axis=0)
            print(data_buffer.times.shape, data_buffer.velocity.shape, data_buffer.data.shape)
        #print("data_buffer.times ", data_buffer.times)
        data_buffer.selected_begin_time=begin_time
        data_buffer.selected_end_time=end_time
        data_buffer.begin_time=data_buffer.times[0]
        data_buffer.end_time=data_buffer.times[len(data_buffer.times)-1]
        data_buffer.bottom_index = datadict['range_index'][0]
        data_buffer.top_index = datadict['range_index'][1]
        data_buffer.param_info=self.param_info
        #print("databuffer ", begin_time, end_time)
        #print("databuffer ", data_buffer.times[0], data_buffer.times[len(data_buffer.times)-1], len(data_buffer.times))
 
        # what is this part for??
        #data_buffer.build_time_index()
        #if self.param_info.range_index != "":
        #    height_map = NcReader.read_1D(matching_files[0], self.param_info.range_index)
        #    data_buffer.build_height_index(height_map)
        #    #print(height_map)
        #    data_buffer.height_map = height_map
        #else:
        #    data_buffer.build_height_index()
        return data_buffer


    def collectMETEO(self, begin_time, end_time, begin_height=0, end_height=0):
        """ """
        data_buffer=DataBuffer.DataBuffer()
        matching_files = self.get_matching_files(begin_time, end_time+3*3600+1) #Jump over last measurement interval
        data_buffer.data=[]
        height_map=[]
                
        for i in range(len(matching_files)):
            data_buffer.times = np.append( data_buffer.times , MeteoReader.time_fom_filename(matching_files[i]))
            data_buffer.data.append(MeteoReader.read_txt(matching_files[i], self.param_info.variable_name))
            height_map.append(MeteoReader.read_txt(matching_files[i], self.param_info.range_dimension_name))


        data_buffer.data=np.array(data_buffer.data)
        height_map=np.array(height_map)

        data_buffer.selected_begin_time=begin_time
        data_buffer.selected_end_time=end_time
        
        data_buffer.begin_time=data_buffer.times[0]
        data_buffer.end_time=data_buffer.times[len(data_buffer.times)-1]
        
        data_buffer.param_info=self.param_info
        data_buffer.build_time_index()
        data_buffer.build_height_index(height_map)
        return data_buffer
    
            
    def read_time(self, filename, begin_index=0, end_index=0):
        """ """
        if self.param_info.system_type in ["Ceilometer_lacros", "Ceilometer_melpitz"]:        
            times=NcReader.read_1D(filename, self.param_info.time_variable_name, begin_index, end_index)
            times -= 2082844800.0
            
        elif self.param_info.system_type == "Cloudnet":
            times=NcReader.read_1D(filename, self.param_info.time_variable_name, begin_index, end_index)            
            times=times.astype(np.double)
            year=int(NcReader.read_int_attribute(filename, "year"))
            month=int(NcReader.read_int_attribute(filename, "month"))
            day=int(NcReader.read_int_attribute(filename, "day"))
            
            dtm=datetime.datetime(year,month,day)
            time_offset=calendar.timegm(dtm.utctimetuple())
            times=times*3600.0
            times=times+time_offset 
 
        #elif self.param_info.system_type == "Polly":
        elif self.param_info.system_type == "Wili" or self.param_info.system_type == "Polly":
            times=[]
            dates=NcReader.read_2D(filename, self.param_info.time_variable_name, begin_index, end_index, 0 , 2)
            for i in range(np.shape(dates)[0]):
                dt=datetime.datetime.strptime(str(dates[i][0]),"%Y%m%d")+datetime.timedelta(seconds=int(dates[i][1]))
                times.append(calendar.timegm(dt.utctimetuple()))
            times=np.array(times)

        elif self.param_info.system_type == "ICON":
            times=NcReader.read_1D(filename, self.param_info.time_variable_name, begin_index, end_index)
            year  = int(NcReader.read_constant(filename,'year'))
            month = int(NcReader.read_constant(filename,'month'))
            day   = int(NcReader.read_constant(filename,'day'))
            dtm=datetime.datetime(year,month,day,0,0,0)
            #dtm=datetime.datetime.strptime("20130425","%Y%m%d")
            date_in_seconds=calendar.timegm(dtm.utctimetuple())
            times=np.array(times)*3600+date_in_seconds
        else :
            times=NcReader.read_1D(filename, self.param_info.time_variable_name, begin_index, end_index)
        
        return times
    

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
        """replaces ``days_available`` and ``day_available``"""
        fh = self.filehandler[which_path]
        groupedby_day = collections.defaultdict(list)
        for d, f in fh:
            groupedby_day[d[0][:8]] += [f]
        no_files_per_day = {k: len(v) for k, v in groupedby_day.items()}
        return no_files_per_day
