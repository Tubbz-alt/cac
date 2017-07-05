#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAC library.

Chip Analysis and Characterization library.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:Licesnse: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 201707
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""
import logging
import numpy as np
import os


class cac(object):
    """The CAC Class Object."""

    # binary data file constants
    HEADER_SZ = 10  # number of 32-bit words in header
    # location of SequenceCount in Header
    # added by streamer
    SEQ_COUNT_OFFST = 4
    # pixel data in framee 68352
    PDATA_SZ = 0  # number of 32-bit words

    def __enter__(self):
        """CAC enter."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """CAC exit."""

    def __init__(self):
        """CAC Class Constructor."""
        self.pBIN = None  # pixel binary file. will be cleared after 2d transformation
        self.img = None  # 2d transformed pixel data
        self.filename = None  # fielname containing binary data
        self.mdate = 0  # last modified date of fielname
        self.asicname = None  # ASIC name

    def epix10ka(self):
        """ePix10a ASIC definitions."""
        self.tot_banks = 4  # total number of banks in a single asic (chip)
        self.cols = 48  # number of columns in bank
        self.tot_cols = self.tot_banks * self.cols  # 192 total columns in chip
        self.tot_rows = 178  # 176 rows + 2 inactive rows
        self.bitmask = 0x00003FFF
        self.asicname = "epix10a"
        self.frame_sz_bytes = 0  # frame size in bytes
        self.tot_frames = 0  # total number of frames in file
        # TODO(abunimeh) tot_chips should be in camera def not ASIC
        self.tot_chips = 4  # total number of chips in cam
        # pixel data in framee 68352 (32-bit words)
        self.PDATA_SZ = int(self.cols / 2 * self.tot_banks * self.tot_chips * self.tot_rows)

    def load(self, filename):
        """Load file content to memory as unsigned 32-bit integer vector.

        Keyword Arguments:
                filename: file object

        Returns:

        """
        # use numpy to load uint32 file
        self.pBIN = np.fromfile(filename, dtype=np.uint32)
        self.filename = os.path.basename(filename)
        self.mdate = os.path.getmtime(filename)

    def save(self):
        """Save 2d pixel content to numpy compressed file.

        Keyword Arguments:

        Returns: False if 2D data doesn't exist, otherwise True

        """
        if self.img is None:
            return False

        logging.debug('saving npz data file...')
        np.savez(self.filename + '.npz', img=self.img, modified=self.mdate,
                 basename=self.filename, asicname=self.asicname)

        return True

    def loadz(self, filename):
        """Load 2D pixel data saved in numpy compressed format.

        Keyword Arguments:
                filename: file object

        Returns: 2D pixel image

        """
        npzfile = np.load(filename)
        self.filename = binfile2d = str(npzfile['basename'])  # binary data file name
        self.asicname = str(npzfile['asicname'])  # asicname
        self.mdate = npzfile['modified']  # last modified
        self.img = npzfile['img']  # pixel image

        if not os.path.exists(binfile2d):
            logging.debug("Couldn't find original dat file, will skip freshness check.")
        else:
            if npzfile['modified'] != os.path.getmtime(binfile2d):
                logging.debug('binary data file has been modified since last extraction, redo...')
                self.load(binfile2d)
                self.img2d()
                if not self.save():
                    logging.error("Couldn't save npz file.")

        return self.img

    def check_epix10ka(self, pBIN=None):
        """Analyze binary data for epix10ka.

        Keyword Arguments:
                pBIN: uint32 binary data from file

        Returns: False if checking fails, otherwise True

        """
        if pBIN is None:
            pBIN = self.pBIN

        # figure out frame size (this is in bytes)
        # assuming that all frames in file have the same exact size # TODO(abunimeh)
        self.frame_sz_bytes = self.pBIN[0]

        # find all frame sizes in file
        frame_szs = self.pBIN[::int(self.frame_sz_bytes / 4) + 1]
        logging.debug('Total Frames: %d ', frame_szs.size)
        if not np.array_equal(frame_szs, (np.full(frame_szs.size, self.frame_sz_bytes, np.uint32))):
            logging.error('frame sizes are not equal, or file is corrupt')
            logging.error(frame_szs)
            return False

        # compensate for additional 32-bit word added by streamer # TODO(abunimeh) this is a bug
        self.frame_sz_bytes = self.frame_sz_bytes + 4  # add 32-bits i.e. 4 bytes
        logging.debug('Frame size: 0x%x words', self.frame_sz_bytes)

        # frame_sz in words (np.uint32)
        frame_sz = int(self.frame_sz_bytes / self.pBIN.itemsize)
        self.tot_frames = frames_total = int(self.pBIN.size / frame_sz)  # different method
        # number of frames
        logging.info('Total number of frames in file: %d', frames_total)

        # quick data integrity check, last SeqCount[31:0] must equal frames_total
        last_frame_index = int((frames_total - 1) * frame_sz)
        last_seq_cnt = self.pBIN[last_frame_index + self.SEQ_COUNT_OFFST]

        first_seq_cnt = self.pBIN[self.SEQ_COUNT_OFFST]

        logging.debug('first sequence count in file: %d', first_seq_cnt)
        logging.debug('last sequence count in file: %d', last_seq_cnt)
        logging.debug('sequence count difference: %d', last_seq_cnt - first_seq_cnt)

        if frames_total != last_seq_cnt:
            logging.warning('Total number of frames in file does not equal SequenceCount')
            if frames_total - 1 != last_seq_cnt - first_seq_cnt:
                logging.warning('Total number of frames does not equal SequenceCount'
                                ' difference. Some frames are missing!')
        return True

    def img2d(self, pBIN=None):
        """Create 2D pixel image from uint32 vector.

        Keyword Arguments:
                pBIN: uint32 binary data from file

        Returns: 2d uint16 numpy array

        """
        if pBIN is None and self.pBIN is None:
            logging.error('pixel binary data is null! Did you load dat file?')
            return None

        pBIN = self.pBIN

        if self.asicname == "epix10a":
            # check if binary data is OK
            if not self.check_epix10ka(pBIN):
                return None
            return self.img2d_epix10ka(pBIN)  # 2D transformation

        logging.debug('asicname is not defined, cannot img2d()')
        return None

    def img2d_epix10ka(self, pBIN):
        """Create 2D pixel image from uint32 vector.

        Note: This function will set pBIN to None to free memory

        Keyword Arguments:
                pBIN: check()ed binary data

        Returns: 2d uint16 numpy array

        """
        # Convert file to uint16 words
        pBIN_16 = self.pBIN.view(np.uint16)
        logging.debug('2D array ' + str(pBIN_16.shape))

        # create 2D array, each row in numpy array is a frame
        pBIN2D = np.reshape(pBIN_16, (-1, int(self.frame_sz_bytes / pBIN_16.itemsize)))
        logging.debug('2D array ' + str(pBIN2D.shape))

        # get rid of header, env data block, and footer
        # HEADER_SZ & PDATA_SZ are  multiplied by 2 because we defined them as 32-bit
        # words, now we are using them on 16-bit words, so they look twice as big
        pBIN2D_pixels = pBIN2D[:, self.HEADER_SZ * 2:self.HEADER_SZ * 2 + self.PDATA_SZ * 2]
        logging.debug('2D pixels only ' + str(pBIN2D_pixels.shape))
        logging.debug('verify last element in first frame...' + str(pBIN2D_pixels[0][-1]))

        # reshape binary data to 3D (FRAMES x WIDTH x HEIGHT), 4 tiled asics
        img = pBIN2D_pixels.reshape(self.tot_frames, self.tot_rows * 2, self.tot_cols * 2)

        img_even = img[:, 0::2, :]  # slice even rows
        img_odd = img[:, 1::2, :]  # slice odd rows

        # move even rows to the bottom i.e. offset by TOTROWS
        img_sorted = np.concatenate((img_odd, img_even), axis=1)
        logging.debug('image shape ' + str(img_sorted.shape))

        self.pBIN = None  # we no longer need raw data
        self.img = img_sorted
        return self.img
