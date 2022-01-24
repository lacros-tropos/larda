#!/usr/bin/python3
# coding=utf-8
""" """

"""
Author: radenz@tropos.de

"""

import sys
sys.path.append('../')
import pyLARDA
import os
from pathlib import Path
import argparse

import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

ROOT_DIR = Path(__file__).absolute().parents[1]
print(ROOT_DIR)

parser = argparse.ArgumentParser(
    description='''
    Example `python3 ListCollector.py -c oceanet_pascal lacros_accept`.
    As default all campaigns are collected.'''
)
parser.add_argument('-c', '--campaign', nargs='+',
                    help='just run for a defined campaign(s)')
args = parser.parse_args()

camp = pyLARDA.LARDA_campaign(ROOT_DIR / Path("../larda-cfg/"), "campaigns.toml")
camp_list = camp.get_campaign_list()
print(camp_list)

if args.campaign:
    assert set(args.campaign).issubset(camp_list), 'campaign not in list'
    camp_list = args.campaign

#larda=pyLARDA.LARDA().connect('test_filepatterns', build_lists=True)

for cname in camp_list:
    larda=pyLARDA.LARDA().connect(cname, build_lists=True)
    print(larda.connectors.keys())
    
#larda=pyLARDA.LARDA('COLRAWI')
#larda=pyLARDA.LARDA('LACROS_at_Leipzig', build_lists=True)
