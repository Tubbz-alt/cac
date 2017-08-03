#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots epix10ka linearity statistics.

Using CAC, plot the average and standard deviation of an image.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:Licesnse: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 201707
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
np.set_printoptions(formatter={'int': hex})

# Align Peaks
APEAK = True

# row = 65
# col = 92
row = 85  # 167 #3
col = 79  # 44

px = np.array([])

pwd = os.getcwd()

pathlist = Path(pwd).glob('**/*.npz')
for path in pathlist:
    p, filename = os.path.split(str(path))
    os.chdir(p)
    npzfile = np.load(filename)
    npx = npzfile['arr_0']
    px = np.vstack([px, npx]) if px.size else npx
    min_px = min(px.size, npx.size)
    os.chdir(pwd)

# px = px[20:23, :]
# Align peaks
if APEAK:
    logging.info("min size is %d", min_px)
    TOP_BTM_DIFF = 5000
    indexes = np.zeros(px.shape[0], dtype=int)
    for r in range(px.shape[0]):
        diffs = np.ediff1d(px[r, :].astype(np.int))
        peak = np.where(abs(diffs) > TOP_BTM_DIFF)[0]
        if peak.shape[0] > 1:  # multiple peaks
            peak = peak[0]  # take first one
        indexes[r] = peak

    # we can pad or roll
    rollval = max(indexes) - indexes
    for r in range(px.shape[0]):
        px[r, :] = np.roll(px[r, :], rollval[r])

px_avg = np.average(px, 0)
px_std = np.std(px, 0)

# Add std deviation bars to the previous plot

plt.figure()
plt.title('%s\n%s runs Linearity Test Pixel (%d,%d), tr1 00' % (filename, str(px.shape[0]),
          row, col))
plt.errorbar(np.arange(min_px), px_avg, yerr=px_std, fmt='--.',  ecolor='r')
plt.show()
plt.plot(px.T)
plt.show()
