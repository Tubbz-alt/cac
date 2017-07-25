#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots noise statistics.

Using CAC, plot the average and standard deviation of an ADC.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:Licesnse: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20170719
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
from cac import cac
import logging
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
    parser.add_argument("-b", "--banks", action="store_true", help="seperate bank plots using gui.")
    parser.add_argument("-e", "--exportplot", action="store_true", help="save plots to svg file.")
    parser.add_argument("-m", "--mask", nargs=1, metavar=('mask'), help="data mask.")
    parser.add_argument("-k", "--skip", nargs=1, metavar=('skip'), type=int, help="skip frames.")
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

            for chip in range(cc.TOT_CHIPS):
                # ASIC arrangement
                #        C0     C1
                # R0   ASIC2 - ASIC1
                # R1   ASIC3 - ASIC0
                if chip == 0:
                    iasic = cc.img[:, cc.tot_rows:, :cc.tot_cols]  # lower left
                elif chip == 1:
                    iasic = cc.img[:, cc.tot_rows:, cc.tot_cols:]  # lower right
                elif chip == 2:
                    iasic = cc.img[:, :cc.tot_rows, cc.tot_cols:]  # upper right
                elif chip == 3:
                    iasic = cc.img[:, :cc.tot_rows, :cc.tot_cols]  # upper left

                # get rid of the 15th bit
                if args.mask:
                    iasic = np.bitwise_and(cc.str2num(args.mask[0]), iasic)

                # grid rid of frames
                if args.skip:
                    iasic = iasic[args.skip[0]:, :, :]

                if args.banks:
                    f, ax = plt.subplots(4, 1)
                    f.suptitle(cc.filename + " ASIC %d" % (chip), fontsize=14)
                    # f.set_size_inches(22, 17)
                    for bank in range(cc.tot_banks):
                        b_data = iasic[:, :, bank*cc.cols:(bank+1)*cc.cols-1]
                        logging.info("ASIC %d bank %d, mu=%f, sigma= %f" % (chip, bank,
                                     np.mean(b_data), np.std(b_data)))
                        # b_avg = np.average(b_data, 0)

                        ax[bank].hist(b_data.ravel(), bins='auto', histtype='step')
                        # ax[bank].plot(b_data.ravel())

                    # plt.tight_layout()
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_banks_plt.png' % (chip))
                    else:
                        plt.show()
                    plt.close()

            logging.info("Done")


if __name__ == "__main__":
    main()
