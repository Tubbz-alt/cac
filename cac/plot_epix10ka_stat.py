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
    parser.add_argument("-p", "--plot", action="store_true", help="plot using gui.")
    parser.add_argument("-e", "--exportplot", action="store_true", help="save plots to svg file.")
    parser.add_argument("-m", "--mask", nargs=1, metavar=('mask'), help="data mask.")
    parser.add_argument("-k", "--skip", nargs=1, metavar=('skip'), type=int, help="skip frames.")
    parser.add_argument("-g", "--singlepixel", nargs=2, type=int, metavar=('row', 'col'),
                        help="plot single pixel across multiple frames.")
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
            if args.mask:
                img_asic0 = np.bitwise_and(cc.str2num(args.mask[0]), img_asic0)

            # grid rid of frames
            if args.skip:
                img_asic0 = img_asic0[args.skip[0]:, :, :]

            img_avg = np.average(img_asic0, 0)  # mean across multiple frames
            img_std = np.std(img_asic0, 0)  # std across multiple frames

            if args.singlepixel:
                # plot single pixel data [frames,y=row,x=col]
                plt.plot(img_asic0[:, args.singlepixel[0], args.singlepixel[1]])
                plt.xlabel('Frame number')
                plt.ylabel('Amplitude')
                plt.title(cc.filename + "\nPixel (%d,%d)" % (args.singlepixel[0],
                                                             args.singlepixel[1]))
                plt.savefig(cc.filename + '_r%dc%d.svg' %
                            (args.singlepixel[0], args.singlepixel[1]))
                plt.show()

            if args.exportplot:
                plt.imshow(img_avg, cmap=cm.plasma)
                # plt.gray()
                plt.colorbar()
                plt.title(cc.filename + '\nImage mean')
                plt.savefig(cc.filename + '_avg.svg')
                # plt.show()
                plt.close()

                plt.hist(img_avg.ravel(), bins='auto', histtype='step')
                plt.title(cc.filename + '\nImage mean histogram')
                plt.savefig(cc.filename + '_mhst.svg')
                # plt.show()
                plt.close()

                plt.imshow(img_std, cmap=cm.plasma)
                # plt.gray()
                plt.colorbar()
                plt.title(cc.filename + '\nImage std')
                plt.savefig(cc.filename + '_std.svg')
                # plt.show()
                plt.close()

                plt.hist(img_std.ravel(), bins='auto', histtype='step')
                plt.title(cc.filename + '\nImage std histogram')
                plt.savefig(cc.filename + '_shst.svg')
                # plt.show()
                plt.close()

            # all in one plot
            if args.plot:
                f, ax = plt.subplots(2, 2)
                im0 = ax[0, 0].imshow(img_avg, cmap=cm.plasma)
                ax[0, 0].set_title('Mean')
                f.colorbar(im0, ax=ax[0, 0])
                ax[0, 1].hist(img_avg.ravel(), bins='auto', histtype='step')
                ax[0, 1].set_title('Mean Histrogram')
                im1 = ax[1, 0].imshow(img_std, cmap=cm.plasma)
                ax[1, 0].set_title('Standard Deviation')
                f.colorbar(im1, ax=ax[1, 0])
                ax[1, 1].hist(img_std.ravel(), bins='auto', histtype='step')
                ax[1, 1].set_title('Standard Deviation histogram')
                # f.subplots_adjust(wspace=0.5, hspace=0.5)
                # f.set_size_inches(11, 8.5)
                # plt.savefig(cc.filename + '_plt.svg')
                plt.tight_layout()
                plt.show()

            logging.info("Done")


if __name__ == "__main__":
    main()
