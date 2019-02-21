#!/usr/bin/python3
# coding=utf-8
""" """

"""
Author: radenz@tropos.de

"""

import pyLARDA
import os

import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)

camp = pyLARDA.LARDA_campaign(ROOT_DIR + "/../larda-cfg/", "campaigns.toml")
camp_list = camp.get_campaign_list()
print(camp_list)

#larda=pyLARDA.LARDA().connect('test_filepatterns', build_lists=True)

for cname in camp_list:
    larda=pyLARDA.LARDA().connect(cname, build_lists=True)
    print(larda.connectors.keys())
    
#larda=pyLARDA.LARDA('COLRAWI')
#larda=pyLARDA.LARDA('LACROS_at_Leipzig', build_lists=True)
