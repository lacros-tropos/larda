#!/usr/bin/python3
# coding=utf-8
""" """

"""
Author: radenz@tropos.de

"""

import pyLARDA

camp = pyLARDA.LARDA_campaign("/home/larda/larda-cfg/", "campaigns.toml")
camp_list = camp.get_campaign_list()
print(camp_list)


for cname in camp_list:
    larda=pyLARDA.LARDA(cname, build_lists=True)
    
#larda=pyLARDA.LARDA('COLRAWI')
#larda=pyLARDA.LARDA('LACROS_at_Leipzig', build_lists=True)
