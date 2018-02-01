#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dump ePix10ka config and pixel map.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20180105
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""
import argparse
import csv
# import logging
import psana as pa

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", help="Show debugging info.")
parser.add_argument("-i", "--index", nargs=1, metavar=('IDX'), type=int, help="camera index.")
parser.add_argument("-r", "--run", nargs=1, metavar=('RUN'), type=int, help="run number.",
                    required=True)

args = parser.parse_args()

# if args.verbose:
#     myloglevel = logging.DEBUG
# else:
#     myloglevel = logging.INFO

run = args.run[0]
if args.index:
    idx = args.index[0]
else:
    idx = 0

# set logging based on args
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=myloglevel)
EXPSTR = 'exp=mfxx32516:run=%d:smd' % (run)
DETECTOR = 'MfxEndstation.0:Epix10ka.%d' % (idx)
# logging.debug("ePix10ka index [%d]", idx)
ED = EXPSTR + "_" + DETECTOR
csvfilename = "".join([c for c in ED if c.isalnum()]).rstrip() + '.csv'

src = pa.Source(DETECTOR)
ds = pa.DataSource(EXPSTR)
env = ds.env()
# get epix config
config = env.configStore().get(pa.Epix.Config10kaV1, src)
if not config:
    print('RUN %s, has no config\n\n' % (run))
else:
    print('\n\n### RUN %s config ###' % (run))
    # traverse config obj
    for c in dir(config):
        # skip anything that starts with _
        if c[0] == "_":
            continue
        # skip asics, we will traverse it later
        if c == "asics":
            continue
        # get attr and check if it is callable
        v = getattr(config, c)
        if not callable(v):
            continue
        # special cases
        if c.lower() == "asicPixelConfigArray".lower():
            try:
                with open(csvfilename, 'w') as mycsvfile:
                    wr = csv.writer(mycsvfile)
                    wr.writerows(v())
            except IOError as e:
                # logging.error("cannot dump to csv file")
                pass
            continue
    # print the value
        print("%s: %s" % (c, v()))

    # print .asics.*
    # using same method above
    for i in range(config.numberOfAsics()):
        print("\tasic%d:" % (i))
        aconf = config.asics(i)
        for c in dir(aconf):
            if c[0] == "_":
                continue
            v = getattr(aconf, c)
            if not callable(v):
                continue
            print("\t\t%s: %s" % (c, v()))

print('\n\nDone')
