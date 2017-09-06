#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots epix10ka linearity fit.

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

x = np.arange(min_px)
y = px[:, :]

# repeat x for all observations
xflat = np.tile(x, px.shape[0])
yflat = y.ravel()  # flatten y-axis for multiple observations

slope, intercept, r_value, p_value, std_err = stats.linregress(xflat, yflat)

logging.info('slope %.3f', slope)
logging.info('intercept %.3f', intercept)
logging.info('R2 %.3f', r_value**2)
logging.info('p %.3f', p_value)
logging.info('std_err %.3f', std_err)

plt.plot(px.T, 'k.')
plt.plot(x, intercept+slope*x, 'r-', linewidth=2)
plt.show()
