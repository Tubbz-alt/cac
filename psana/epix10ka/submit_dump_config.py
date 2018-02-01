#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dump ePix10ka config and pixel map. Submit result in JSON to LCLS ARP
see https://confluence.slac.stanford.edu/display/PSDM/Automatic+Run+Processing

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20180105
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
import csv
import difflib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import psana as pa
import requests
import sys
mpl.use('Agg')


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map."""
    # see https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


update_url = os.environ.get('BATCH_UPDATE_URL')
OUTURL = "https://pswww.slac.stanford.edu/experiment_results/MFX/1107-mfxx32516/configs/"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--diff", action="store_true", help="diff with last run.")
parser.add_argument("-p", "--post", action="store_true", help="post result to log.")
parser.add_argument("-i", "--index", nargs=1, metavar=('IDX'), type=int, help="camera index.")
parser.add_argument("-o", "--output", nargs=1, metavar=('FILE'), help="dump to file.")
parser.add_argument("-r", "--run", nargs=1, metavar=('RUN'), type=int, help="run number.",
                    required=True)

args = parser.parse_args()

run = args.run[0]
if args.index:
    idx = args.index[0]
else:
    idx = 0

# set logging based on args
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=myloglevel)
# EXPSTR = 'exp=mfxx32516:run=%d:smd:live' % (run)
EXPSTR = 'exp=mfxx32516:run=%d:smd:dir=/reg/d/ffb/mfx/mfxx32516/xtc:live' % (run)
DETECTOR = 'MfxEndstation.0:Epix10ka.%d' % (idx)
# logging.debug("ePix10ka index [%d]", idx)
ED = EXPSTR + "_" + DETECTOR

if args.output:
    ofile = args.output[0]
    sys.stdout = open(ofile, "w+")
else:
    # ofile = "".join([c for c in ED if c.isalnum()]).rstrip()
    ofile = 'r%d_%d.conf' % (run, idx)

src = pa.Source(DETECTOR)
ds = pa.DataSource(EXPSTR)
det = pa.Detector(DETECTOR)

# wait until we have some data, otherwise config will not exist
for evt in ds.events():
    if det.raw(evt) is not None:
        break

env = ds.env()
# get epix config
postit = 1
config = env.configStore().get(pa.Epix.Config10kaV1, src)
if not config:
    print('RUN %s, has no config\n\n' % (run))
    postit = 0
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
            parr = v()
            # im = Image.fromarray(parr)
            # im.save(ofile + '.jpg')
            fig, ax = plt.subplots()
            fig.set_dpi(150)
            fig.set_size_inches(8, 6)
            cax = ax.imshow(parr, vmin=0, vmax=16, cmap=discrete_cmap(16, 'Paired'))
            ax.set_title('Pixel map run# %d' % (run))
            cbar = fig.colorbar(cax, ticks=range(16))
            cbar.ax.set_yticklabels(np.arange(0, 16, 1))
            fig.savefig(ofile + '.png')
            # scipy.misc.imsave(ofile + '.png', parr)
            try:
                with open(ofile + '.csv', 'w') as mycsvfile:
                    wr = csv.writer(mycsvfile)
                    wr.writerows(parr)
            except IOError as e:
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

if args.output:
    sys.stdout.flush()
    sys.stdout.close()

if args.diff:
    diffexists = False
    lrun = int(run)-1  # last run
    dirc, bfile = os.path.split(ofile)
    fromfile = dirc + '/r%d_%d.conf' % (lrun, idx)
    tofile = ofile
    if os.path.exists(fromfile) and os.path.exists(tofile):
        fromlines = open(fromfile, 'r').readlines()
        tolines = open(tofile, 'r').readlines()
        diff = difflib.HtmlDiff().make_file(fromlines, tolines, fromfile, tofile)
        try:
            with open(ofile + "_diff.html", 'w') as difffile:
                difffile.write(diff)
                diffexists = True
        except IOError as e:
            pass

if args.post:
    if postit:
        pngurl = OUTURL + 'r%d_%d.conf.png' % (run, idx)
        csvurl = OUTURL + 'r%d_%d.conf.csv' % (run, idx)
        r = requests.post(update_url,
                          json={'counters': {'Epix10ka%d' % (idx): ['config dumped', 'green'],
                                'Mask%d' % (idx):
                                '<a href="' + pngurl + '">png</a>'
                                ' <a href="' + csvurl + '">csv</a>'}})
        if args.diff:
            if diffexists:
                shortfile = 'r%d_%d.conf_diff.html' % (run, idx)
                diffurl = OUTURL + shortfile
                r = requests.post(update_url,
                                  json={'counters': {'diff %d' % (idx): '<a href="' + diffurl
                                        + '">view diff</a>'}})
            else:
                r = requests.post(update_url,
                                  json={'counters': {'diff %d' % (idx): 'NA'}})
    else:
        r = requests.post(update_url,
                          json={'counters': {'Epix10ka%d' % (idx): ['no config', 'red']}})
    # print(r)
