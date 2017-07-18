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

from cac import cac
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
np.set_printoptions(formatter={'int': hex})

# create cac object
with cac() as cc:

    cc.epix10ka()  # set epix10ka params
    row = '3'
    col = '80'
    filename = 'linearityRow_10ka_120Hz_hrtest_false_row_' + row
    cc.loadz(filename + '_run2.dat.npz')
    img2_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]
    img2_asic0 = np.bitwise_and(0x3fff, img2_asic0)

    cc.loadz(filename + '_run3.dat.npz')
    img3_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]
    img3_asic0 = np.bitwise_and(0x3fff, img3_asic0)

    cc.loadz(filename + '_run4.dat.npz')
    img4_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]
    img4_asic0 = np.bitwise_and(0x3fff, img4_asic0)

    cc.loadz(filename + '_run5.dat.npz')
    img5_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]
    img5_asic0 = np.bitwise_and(0x3fff, img5_asic0)

    cc.loadz(filename + '_run6.dat.npz')
    img6_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]
    img6_asic0 = np.bitwise_and(0x3fff, img6_asic0)

    px2 = img2_asic0[:, int(row), int(col)]
    px3 = img3_asic0[:, int(row), int(col)]
    px4 = img4_asic0[:, int(row), int(col)]
    px5 = img5_asic0[:, int(row), int(col)]
    px6 = img6_asic0[:, int(row), int(col)]

    min_px = min(px2.size, px3.size)
    min_px = min(min_px, px4.size)
    min_px = min(min_px, px5.size)
    min_px = min(min_px, px6.size)

    logging.info('min size is %d' % (min_px))

    # create statistics
    px = np.vstack((px2[:min_px], px3[:min_px], px4[:min_px], px5[:min_px], px6[:min_px]))
    px_avg = np.average(px, 0)
    px_std = np.std(px, 0)

    plt.title(filename + '\n5 run Linearity Test Pixel ('+row+','+col+'), tr1 00')
    plt.xlabel('Frame number')
    plt.ylabel('Amplitude')
    plt.plot(np.arange(min_px), px_avg, 'k')
    plt.fill_between(np.arange(min_px), px_avg-px_std/2, px_avg+px_std/2, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(filename + '_lin.svg')
    plt.show()
