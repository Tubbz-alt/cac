#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots epix10ka linearity statistics.

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
from scipy import stats

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
np.set_printoptions(formatter={'int': hex})

# Align Peaks
APEAK = True

# row = 65
# col = 92
row = 10  # 167 #3
col = 10  # 44
ADCRESOL = 14  # bits

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

logging.info("min px size is %d", min_px)
plt.plot(px.T, '-')
plt.show()

# px = px[20:23, :]
# TOP_BTM_DIFF = 230 # tr0
TOP_BTM_DIFF = 1000  # tr1
indexes = np.zeros(px.shape[0], dtype=int)
for r in range(px.shape[0]):
    diffs = np.ediff1d(px[r, :].astype(np.int))
    peak = np.where(abs(diffs) > TOP_BTM_DIFF)[0]
    if peak.shape[0] > 1:  # multiple peaks
        peak = peak[0]  # take first one
    indexes[r] = peak

# Align peaks
if APEAK:
    # we can pad or roll
    rollval = max(indexes) - indexes
    for r in range(px.shape[0]):
        px[r, :] = np.roll(px[r, :], rollval[r])

# find falling edge
TOP_BTM_DIFF = 5000
indexes = np.zeros(px.shape[0], dtype=int)
for r in range(px.shape[0]):
    diffs = np.ediff1d(px[r, :].astype(np.int))
    peak = np.where(abs(diffs) > TOP_BTM_DIFF)[0]
    if peak.shape[0] > 1:  # multiple peaks
        peak = peak[0]  # take first one
    indexes[r] = peak

# plot average across multiple runs
plt.title('%s\n%s runs Linearity Test Pixel (%d,%d), tr1 00' % (filename, str(px.shape[0]), row,
                                                                col))
px_avg = np.average(px, 0)
px_std = np.std(px, 0)
plt.errorbar(np.arange(min_px), px_avg, yerr=px_std, fmt='--.', ecolor='r')
plt.plot(px.T, '-')
plt.show()

# plot poly fits
# create axis
x = np.arange(min_px)
# rampstart = 150 # tr0
# rampstart = 660  # tr1
# rampstart = 150
rampstart = 60
# rampstop = max(indexes) - 12
rampstop = 288
# fitstart = max(indexes)+1
# fitstart = 870
# fitstart = rampstop + 24
fitstart = 300

# tail after autoranging switch. we are only interested in this range
rx = x[rampstart:rampstop]
ry = px[:, rampstart:rampstop]

# repeat x for all observations
rxflat = np.tile(rx, px.shape[0])
ryflat = ry.ravel()  # flatten y-axis for multiple observations

# coefficients, highest power first
rcoeff = np.polyfit(rxflat, ryflat, 1)  # first order polyfit or ramp
rfit1 = np.poly1d(rcoeff)

slope, intercept, r_value, p_value, std_err = stats.linregress(rxflat, ryflat)
print(slope)
print(intercept)
print(r_value**2)
print(p_value)
print(std_err)

# tail after autoranging switch. we are only interested in this range
tx = x[fitstart:]
ty = px[:, fitstart:]

# repeat x for all observations
txflat = np.tile(tx, px.shape[0])
tyflat = ty.ravel()  # flatten y-axis for multiple observations

tcoeff = np.polyfit(txflat, tyflat, 1)  # first order polyfit for tail
tfit1 = np.poly1d(tcoeff)
slope, intercept, r_value, p_value, std_err = stats.linregress(txflat, tyflat)
print(slope)
print(intercept)
print(r_value**2)
print(p_value)
print(std_err)

plt.plot(px.T, 'k.')
plt.plot(rx, rfit1(rx), '-', linewidth=3)
plt.plot(tx, tfit1(tx), '-', linewidth=3)

# data transformation
rtheta = np.arctan(rcoeff[0])
ttheta = np.arctan(tcoeff[0])
k = rcoeff[0] / tcoeff[0]
yoff = rfit1(rampstop)
tintercept = -1 * tcoeff[1] / tcoeff[0]

newtxflat = (txflat - rampstop) * np.cos(rtheta) / np.cos(ttheta) + rampstop
newtyflat = (tyflat - tfit1(rampstop)) * np.sin(rtheta) / np.sin(ttheta) + rfit1(rampstop)

plt.plot(newtxflat, newtyflat, '.', color='#808080')

# rotation matrix
# rotatetheta = rtheta - ttheta
# c, s = np.cos(rotatetheta), np.sin(rotatetheta)
# R = np.matrix([[c, -s], [s, c]])

# replot scaled fit
tlen = tx.size / np.cos(ttheta)
newtx_len = np.cos(rtheta) * tlen
newtx = rampstop + np.arange(newtx_len)

plt.plot(newtx, rfit1(newtx), '-', linewidth=3)
plt.show()

# INL plots
ryfit = np.polyval(rcoeff, rxflat)
tyfit = np.polyval(tcoeff, txflat)

plt.plot(rxflat - min(rxflat), 100 * (ryfit - ryflat) / (max(ryflat) - min(ryflat)), '.')
plt.title('High Gain INL')
plt.show()

plt.plot(txflat - min(txflat), 100 * (tyfit - tyflat) / (2**ADCRESOL - min(tyflat)), '.')
plt.title('Low Gain INL')
plt.show()

print(rfit1)
print(tfit1)
# print the ratio between the slopes
print("ratio is %f" % (rcoeff[0] / tcoeff[0]))
