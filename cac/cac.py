#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAC library.

Chip Analysis and Characterization library.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
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
    # location of SequenceCount in Header
    # added by streamer
    SEQ_COUNT_OFFST = 4
    # pixel data in frame 68352
    PDATA_SZ = 0  # number of 32-bit words
    # total number of chips in cam
    TOT_CHIPS = 4

    def __enter__(self):
        """CAC enter."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """CAC exit."""

    def __init__(self):
        """CAC Class Constructor."""
        self.pBIN = None  # pixel binary file. will be cleared after 3D transformation
        self.img = None  # 3D transformed pixel data
        self.filename = None  # fielname containing binary data
        self.mdate = 0  # last modified date of fielname
        self.asicname = None  # ASIC name
        self.asic = None  # 3D transformed pixel data of a single ASIC

    def epix10ka(self):
        """ePix10a ASIC definitions."""
        self.asicname = "epix10a"
        self.tot_banks = 4  # total number of banks in a single asic (chip)
        self.cols = 48  # number of columns in bank
        self.tot_cols = self.tot_banks * self.cols  # 192 total columns in chip
        self.tot_rows = 178  # 176 rows + 2 inactive rows
        self.bitmask = 0x00003FFF
        # SuperRows are defined by Data Streamer. i.e. CameraRow i.e. a row of two ASICS
        self.superrow = (self.tot_cols * self.TOT_CHIPS / 2) / 2  # two columns per word
        self.header_sz = 10  # number of 32-bit words in header
        self.envdata_sz = 384  # number of words in env data block
        self.tps_sz = 2  # number of words in env data block
        self.footer_sz = 1  # number of words in env data block
        self.frame_sz = self.header_sz - 1 + 2 * self.tot_rows * self.superrow + \
            self.envdata_sz + self.tps_sz + self.footer_sz  # frame size in bytes
        self.frame_sz_bytes = int(self.frame_sz*4)  # frame size in bytes

        self.tot_frames = 0  # total number of frames in file
        # pixel data in framee 68352 (32-bit words)
        self.PDATA_SZ = int(self.cols / 2 * self.tot_banks * self.TOT_CHIPS * self.tot_rows)

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

    def save(self, asic=None):
        """Save 3D pixel content to numpy compressed file.

        Keyword Arguments:

        Returns: False if 3D data doesn't exist, otherwise True

        """
        if self.img is None:
            return False

        logging.debug('saving npz data file...')
        if asic is None:
            np.savez(self.filename + '.npz', img=self.img, modified=self.mdate,
                     basename=self.filename, asicname=self.asicname)
        else:
            self.getasic_epix10ka(asic)
            np.savez(self.filename + '_' + str(asic) + '.npz', asic=self.asic, modified=self.mdate,
                     basename=self.filename, asicname=self.asicname)

        return True

    def loadz(self, filename):
        """Load 3D pixel data saved in numpy compressed format.

        Keyword Arguments:
                filename: file object

        Returns: 3D pixel image

        """
        npzfile = np.load(filename)
        self.filename = binfile2d = str(npzfile['basename'])  # binary data file name
        self.asicname = str(npzfile['asicname'])  # asicname
        self.mdate = npzfile['modified']  # last modified
        if 'img' in npzfile:
            self.img = npzfile['img']  # pixel image
            if not os.path.exists(binfile2d):
                logging.debug("Couldn't find original dat file, will skip freshness check.")
            else:
                if npzfile['modified'] != os.path.getmtime(binfile2d):
                    logging.debug('Data file has been modified since last extraction, redo...')
                    self.load(binfile2d)
                    self.img2d()
                    if not self.save():
                        logging.error("Couldn't save npz file.")

            return self.img
        if 'asic' in npzfile:
            self.asic = npzfile['asic']  # pixel image
            return self.asic

    def check_epix10ka(self, pBIN=None):
        """Analyze binary data for epix10ka.

        Keyword Arguments:
                pBIN: uint32 binary data from file

        Returns: False if checking fails, otherwise True

        """
        if pBIN is None:
            pBIN = self.pBIN

        # verify file sequence i.e. integrity
        # 1st word is defined as a the frame size in bytes
        if not self.verify_seq_epix10ka(self.pBIN, self.pBIN[0]):
            logging.error('data failed verification!')
            return False

        # compensate for additional 32-bit word added by streamer
        # TODO(abunimeh) this is a bug. Value should be all inclusive!
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
            frame_seqs = self.pBIN[self.SEQ_COUNT_OFFST::frame_sz]
            np.savetxt(self.filename + '_seqs.csv', frame_seqs, delimiter='\n', fmt='%u')
            if frames_total - 1 != last_seq_cnt - first_seq_cnt:
                logging.warning('Total number of frames does not equal SequenceCount'
                                ' difference. Some frames are missing!')
        return True

    def verify_seq_epix10ka(self, pBIN=None, seq=None, itr=0):
        """Analyze binary data for epix10ka.

        Note: This function will quickly return True if the file is not corrupt.
        If there is a single frame in the middle of the data file the speed is relatively fast.
        If the first frame has wrong size, things will slow down. The more corrupt frames the file
        has the slower this function will be.

        Keyword Arguments:
                pBIN: uint32 binary data from file
                seq: word repeated multiple times in file.
                itr: number of iterations it took me to return()

        Returns: False if checking fails, otherwise True

        """
        while True:
            logging.debug("checking data [%d]...", itr)

            if seq is None:
                logging.error("Missing frame size.")
                return False

            # seq is defined in bytes, convert it to words then add one for uncounted 1st word
            step = int(seq / 4) + 1
            # verify if we have a valid frame size
            if step >= self.pBIN.size:
                logging.error("File is corrupt. Frame size is bigger than file size.")
                return False

            if step <= self.header_sz:
                logging.error("File is corrupt. Frame size is smaller than header.")
                return False

            if seq != self.frame_sz_bytes:
                logging.warning('frame [0x%x] size is 0x%x instead of 0x%x, trying to skip,...',
                                itr, seq, self.frame_sz_bytes)
                itr += 1
                self.pBIN = self.pBIN[step:]
                seq = self.pBIN[0]
                continue

            frame_szs = self.pBIN[::step]  # quickly get all frame sizes
            # find odd frame sizes, if any
            findx = np.array_equal(frame_szs, (np.full(frame_szs.size, seq, np.uint32)))
            if not findx:
                logging.warning('frame sizes are not equal, trying to skip,...')
                # compare expected seq with actual seq by subtraction
                invalid = frame_szs - seq

                invalidind = invalid.nonzero()[0]  # first occurrence
                logging.warning("Skip offending frame: [0x%x]", invalidind[0])
                logging.debug("Frame Offset in file: [0x%x]", invalidind[0] * (seq + 4))

                offset = invalidind[0] * step  # actual offset in array
                length = int(self.pBIN[offset] / 4)  # length in words

                if offset > self.pBIN.size or offset + length + 1 > self.pBIN.size:
                    logging.error("File is corrupt. Frame size is bigger than file size.")
                    return False

                logging.debug("Skipping Frame#[0x%x], len=[0x%x], size=[0x%x] ", invalidind[0],
                              length, self.pBIN[offset])

                adjpBIN = self.pBIN[:offset]  # slice until offending frame
                logging.debug("last valid frame footer : 0x%x", adjpBIN[-1])
                skippBIN = self.pBIN[offset + length + 1:]
                if offset + length + 1 < self.pBIN.size:
                    logging.debug("next frame word: 0x%x", skippBIN[0])

                self.pBIN = np.concatenate((adjpBIN, skippBIN))  # combine to skip offending frame

                logging.debug("removed offending frame, trying again...")
                itr += 1
                continue
            break

        logging.debug("Number of iterations to reconstruct file: [%d]...", itr)
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
            if not self.check_epix10ka(self.pBIN):
                return None
            return self.img2d_epix10ka(self.pBIN)  # 2D transformation

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
        logging.debug('16-bit array ' + str(pBIN_16.shape))

        # create 2D array, each row in numpy array is a frame
        pBIN2D = np.reshape(pBIN_16, (-1, int(self.frame_sz_bytes / pBIN_16.itemsize)))
        logging.debug('2D array ' + str(pBIN2D.shape))

        # get rid of header, env data block, and footer
        # header_sz & PDATA_SZ are  multiplied by 2 because we defined them as 32-bit
        # words, now we are using them on 16-bit words, so they look twice as big
        pBIN2D_pixels = pBIN2D[:, self.header_sz * 2:self.header_sz * 2 + self.PDATA_SZ * 2]
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

    def str2num(self, x):
        """Convert a str decimal or hex to integer.

        Keyword Arguments:
                x: string number

        Returns: integer

        """
        # default is integers
        base = 10
        # convert hex strings to hex values
        if '0x' in x:
            base = 16
        return int(x, base)

    def getasic_epix10ka(self, index):
        """Get single ASIC data and return reference.

        Keyword Arguments:
                index: ASIC number

        Returns: Numpy Array containing ASIC data

        """
        if index == 0:
            self.asic = self.img[:, self.tot_rows:, self.tot_cols:]  # lower right
        elif index == 1:
            self.asic = self.img[:, :self.tot_rows, self.tot_cols:]  # upper right
        elif index == 2:
            self.asic = self.img[:, :self.tot_rows, :self.tot_cols]  # upper left
        elif index == 3:
            self.asic = self.img[:, self.tot_rows:, :self.tot_cols]  # lower left
        else:
            return None

        return self.asic
