#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots epix10ka statistics.

Using CAC, plot the average and standard deviation of an image.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:Licesnse: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 201707
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
from cac import cac
import logging
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy.stats import norm
import sys


def main():
    """Routine to plot data. This uses argparse to get cli params.

    Example script using CAC library.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debugging info.")
    parser.add_argument("-s", "--save", action="store_true", help="save npz array.")
    parser.add_argument("-p", "--plot", action="store_true", help="plot all in one using gui.")
    parser.add_argument("-n", "--nplots", action="store_true", help="seperate plots using gui.")
    parser.add_argument("-b", "--banks", action="store_true", help="seperate bank plots using gui.")
    parser.add_argument("-l", "--baseline", action="store_true", help="baseline only.")
    parser.add_argument("-e", "--exportplot", action="store_true", help="save plots to svg file.")
    parser.add_argument("-m", "--mask", nargs=1, metavar=('mask'), help="data mask.")
    parser.add_argument("-r", "--rmrow", nargs=1, metavar=('rmrow'), type=int, help="remove rows.")
    parser.add_argument("-c", "--rmcol", nargs=1, metavar=('rmrow'), type=int, help="remove cols.")
    parser.add_argument("-k", "--skip", nargs=1, metavar=('skip'), type=int, help="skip frames.")
    parser.add_argument("-a", "--asic", nargs=1, metavar=('asic'), type=int, help="asic index.")
    parser.add_argument("-g", "--singlepixel", nargs=2, type=int, metavar=('row', 'col'),
                        help="plot single pixel across multiple frames.")
    parser.add_argument("-x", "--extractpixel", nargs=4, type=int,
                        metavar=('row', 'col', 'left', 'right'),
                        help="extract pixel data across multiple frames.")
    parser.add_argument("-f", "--fitpixel", nargs=2, type=int,
                        metavar=('left', 'right'),
                        help="extract pixel data across multiple frames.")

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
            logging.debug('image shape ' + str(cc.img.shape))

            # save npz data after analysis
            if args.save:
                cc.save()

            for chip in range(cc.TOT_CHIPS):
                # ASIC arrangement
                #     ___C0_____C1_____
                #    |                |
                # R0 |  ASIC2 - ASIC1 |
                # R1 |  ASIC3 - ASIC0 |
                #    |________________|

                if args.asic:
                    if args.asic[0] != chip:
                        # skip other asics
                        continue

                if chip == 0:
                    iasic = cc.img[:, cc.tot_rows:, cc.tot_cols:]  # lower right
                elif chip == 1:
                    iasic = cc.img[:, :cc.tot_rows, cc.tot_cols:]  # upper right
                elif chip == 2:
                    iasic = cc.img[:, :cc.tot_rows, :cc.tot_cols]  # upper left
                elif chip == 3:
                    iasic = cc.img[:, cc.tot_rows:, :cc.tot_cols]  # lower left
                # get rid of the 15th bit
                if args.mask:
                    iasic = np.bitwise_and(cc.str2num(args.mask[0]), iasic)

                # grid rid of frames
                if args.skip:
                    iasic = iasic[args.skip[0]:, :, :]

                # grid rid of rows from bottom
                if args.rmrow:
                    iasic = iasic[:, :cc.tot_rows-args.rmrow[0], :]

                # grid rid of cols. This will delete pairs
                if args.rmcol:
                    iasic = np.delete(iasic, (args.rmcol[0], args.rmcol[0]+1), axis=2)

                # remove all data except baseline
                if args.baseline:
                    iasic = iasic[:, cc.tot_rows-2:cc.tot_rows-1, :]

                img_avg = np.average(iasic, 0)  # mean across multiple frames
                img_std = np.std(iasic, 0)  # std across multiple frames

                if args.extractpixel:
                    # px = iasic[1000:2100, args.singlepixel[0], args.singlepixel[1]]
                    px = iasic[args.extractpixel[2]:args.extractpixel[3],
                               args.extractpixel[0], args.extractpixel[1]]
                    logging.debug('saving npz data file...')
                    np.savez("%s_%d_%d%s" % (filename, args.extractpixel[0], args.extractpixel[1],
                             '.npz'), px)

                if args.singlepixel:
                    # plot single pixel data [frames,y=row,x=col]
                    spdata = iasic[:, args.singlepixel[0], args.singlepixel[1]]
                    plt.plot(spdata)
                    plt.xlabel('Frame number')
                    plt.ylabel('Amplitude')
                    plt.title(cc.filename + "ASIC %d" % (chip) +
                              "\nPixel (%d,%d)" % (args.singlepixel[0],
                                                   args.singlepixel[1]))

                    if args.fitpixel:
                        px = iasic[args.fitpixel[0]:args.fitpixel[1],
                                   args.singlepixel[0], args.singlepixel[1]]

                        x = np.linspace(args.fitpixel[0], args.fitpixel[1], px.shape[0])
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x,
                                                                                       px.ravel())

                        logging.info('slope %.3f', slope)
                        logging.info('intercept %.3f', intercept)
                        logging.info('R2 %.3f', r_value**2)
                        logging.info('p %.3f', p_value)
                        logging.info('std_err %.3f', std_err)

                        plt.plot(x, intercept+slope*x, 'r-', linewidth=2)

                    if args.exportplot:
                        plt.savefig(cc.filename + '_r%dc%d.png' %
                                    (args.singlepixel[0], args.singlepixel[1]))
                    else:
                        plt.show()

                    # plot histogram of pixel data without averaged darkframe
                    # n, bins, patches = plt.hist(spdata-np.average(spdata), bins=32,
                    #                             histtype='step')
                    n, bins, patches = plt.hist(spdata-np.average(spdata), bins=64)
                    plt.title(cc.filename + "ASIC %d" % (chip) +
                              "\nPixel (%d,%d) Histogram" % (args.singlepixel[0],
                                                             args.singlepixel[1]))
                    # Gaussian fit
                    ax = plt.gca()
                    (mu, sigma) = norm.fit(spdata.ravel()-np.average(spdata))
                    plt.text(1., 1., r'$\mu=%.3f,\ \sigma=%.3f$' % (mu, sigma),
                             transform=ax.transAxes, horizontalalignment='right',
                             verticalalignment='top')
                    normscale = np.sum(n * np.diff(bins))
                    y = mlab.normpdf(bins, mu, sigma) * normscale  # scale it to match height
                    plt.plot(bins, y, 'r--')
                    plt.xlabel('ADC Count')
                    plt.ylabel('Frequency')
                    if args.exportplot:
                        plt.savefig(cc.filename + '_r%dc%d_hst.png' %
                                    (args.singlepixel[0], args.singlepixel[1]))
                    else:
                        plt.show()

                # seperate flot for each waveform
                if args.nplots:
                    plt.imshow(img_avg, cmap=cm.plasma)
                    plt.colorbar()
                    plt.tight_layout()
                    plt.title(cc.filename + '\nASIC %d Image mean' % (chip))
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_avg.svg' % (chip))
                    else:
                        plt.show()
                    plt.close()

                    plt.hist(img_avg.ravel(), bins='auto', histtype='step')
                    plt.title(cc.filename + '\nASIC %d Image mean histogram' % (chip))
                    plt.tight_layout()
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_mhst.svg' % (chip))
                    else:
                        plt.show()
                    plt.close()

                    plt.imshow(img_std, cmap=cm.plasma)
                    plt.colorbar()
                    plt.tight_layout()
                    plt.title(cc.filename + '\nASIC %d Image std' % (chip))
                    if args.exportplot:
                        plt.savefig(cc.filename + '_std.svg')
                    else:
                        plt.show()
                    plt.close()

                    plt.hist(img_std.ravel(), bins='auto', histtype='step')
                    plt.title(cc.filename + '\nASIC %d Image std histogram' % (chip))
                    plt.tight_layout()
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_shst.svg' % (chip))
                    else:
                        plt.show()
                    plt.close()

                # all in one plot
                if args.plot:
                    f, ax = plt.subplots(2, 2)
                    f.suptitle(cc.filename + " ASIC %d" % (chip), fontsize=14)
                    f.set_size_inches(11, 8.5)
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
                    # plt.tight_layout()
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_plt.svg' % (chip))
                    else:
                        plt.show()
                    plt.close()

                if args.banks:
                    f, ax = plt.subplots(4, 4)
                    f.suptitle(cc.filename + " ASIC %d" % (chip), fontsize=14)
                    f.set_size_inches(22, 17)
                    for bank in range(cc.tot_banks):
                        b_avg = np.average(iasic[:, :, bank*cc.cols:(bank+1)*cc.cols-1], 0)
                        b_std = np.std(iasic[:, :, bank*cc.cols:(bank+1)*cc.cols-1], 0)

                        im0 = ax[bank, 0].imshow(b_avg, cmap=cm.plasma)
                        # ax[bank, 0].set_title('Bank %d Mean' % (bank))
                        f.colorbar(im0, ax=ax[bank, 0])
                        ax[bank, 1].hist(b_avg.ravel(), bins='auto', histtype='step')
                        # ax[bank, 1].set_title('Bank %d Mean Histrogram' % (bank))
                        im1 = ax[bank, 2].imshow(b_std, cmap=cm.plasma)
                        # ax[bank, 2].set_title('Bank %d Standard Deviation' % (bank))
                        f.colorbar(im1, ax=ax[bank, 2])
                        ax[bank, 3].hist(b_std.ravel(), bins='auto', histtype='step')
                        # ax[bank, 3].set_title('Bank %d Standard Deviation histogram' % (bank))

                    plt.tight_layout()
                    if args.exportplot:
                        plt.savefig(cc.filename + '_%d_banks_plt.png' % (chip))
                    else:
                        plt.show()
                    plt.close()

            logging.info("Done")


if __name__ == "__main__":
    main()
