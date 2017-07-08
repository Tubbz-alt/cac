#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots epix10ka statistics.

Using CAC, plot the average and standard deviation of an image.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:Licesnse: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 201707
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
from cac import cac
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def main():
    """Routine to plot data. This uses argparse to get cli params.

    Example script using CAC library.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debugging info.")
    parser.add_argument("-s", "--save", action="store_true", help="save npz array.")
    parser.add_argument("-i", "--inputfile", nargs=1, metavar=('FILE'),
                        help="File name to plot.", required=True)
    args = parser.parse_args()

    # show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        myloglevel = logging.DEBUG
    else:
        myloglevel = logging.INFO

    # set logging based on args
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=myloglevel)
    np.set_printoptions(formatter={'int': hex})

    # create cac object
    with cac() as cc:
        if args.inputfile:
            cc.epix10ka()  # set epix10ka params

            filename = args.inputfile[0]
            if not os.path.isfile(filename):
                logging.error("[%s] does not exists!", filename)
                sys.exit(1)
            filext = filename.split(".")[-1]
            if filext == "dat":
                logging.info("[%s] loading raw data file", cc.asicname)
                cc.load(filename)
                cc.img2d()
            elif filext == "npz":
                logging.info("[%s] loading compressed data file", cc.asicname)
                cc.loadz(filename)
            else:
                logging.error("[%s] file extension is not recognized!", filext)
                sys.exit(1)

            # check if loading data is OK
            if cc.img is None:
                logging.error("Binary data is missing!")
                sys.exit(1)

            # save npz data after analysis
            if args.save:
                cc.save()

            # ASIC arrangement
            #        C0     C1
            # R0   ASIC2 - ASIC1
            # R1   ASIC3 - ASIC0
            img_asic0 = cc.img[:, cc.tot_rows:, cc.tot_cols:]  # lower right
            # img_asic1 = cc.img[:, :cc.tot_rows, cc.tot_cols:]  # upper right
            # img_asic2 = cc.img[:, :cc.tot_rows, :cc.tot_cols]  # upper left
            # img_asic3 = cc.img[:, :cc.tot_rows, cc.tot_cols:]  # lower left

            # get rid of the 15th bit
            # img_asic0 = np.bitwise_and(0x3FFF, img_asic0)
            # img_asic0 = img_asic0[100:, :, :]  # skip first 10 frames

            # plot single pixel data [frames,y=row,x=col]
            # plt.plot(img_asic0[:, 150, 136])
            # plt.xlabel('frame number')
            # plt.ylabel('amplitude')
            # plt.title('Single Pix')
            # plt.show()

            img_avg = np.average(img_asic0, 0)  # mean across multiple frames
            img_std = np.std(img_asic0, 0)  # std across multiple frames

            plt.imshow(img_avg, cmap=cm.plasma)
            # plt.gray()
            plt.colorbar()
            plt.title('Image mean [' + cc.filename + ']')
            plt.savefig(cc.filename + '_avg.svg')
            # plt.show()

            plt.hist(img_avg.ravel(), bins='auto')  # arguments are passed to np.histogram
            plt.title('Image Mean Histrogram [' + cc.filename + ']')
            plt.savefig(cc.filename + '_hst.svg')
            # plt.show()

            plt.imshow(img_std, cmap=cm.plasma)
            # plt.gray()
            plt.colorbar()
            plt.title('Image std [' + cc.filename + ']')
            plt.savefig(cc.filename + '_std.svg')
            # plt.show()

            logging.info("Done")


if __name__ == "__main__":
    main()
