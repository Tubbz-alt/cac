#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
epixhr test structure data plot

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20180629
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

# import h5py
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import os
import logging
import sys

# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

np.set_printoptions(formatter={'int': hex})
fig_id=0

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
   fname = 'test.dat'

#------ uint 16 ------
# Packet Size [31:16] 
# Packet Size [15:0]  
# Flags [31:16]       
# Flags [15:0]        
# x00 & Lane Number & VC Number
# x0000 
# Acquisition number [15:0]
# Acquisition number [31:16]
# x000 & 0 & ASIC number
# x0000
# Sample 0
# Sample 1
# ...
# Sample n (i.e. NSAMPLES)
#------ uint 16 ------

NSAMPLES = 30  # Number of ADC Samples in a packet
HEADER_SZ = 10  # number of 16-bit words in header
PKT_SZ = (HEADER_SZ + NSAMPLES) * 2  # Size of epixhr ts packet in bytes
PKT_WRD = NSAMPLES + HEADER_SZ  # number of words in a single packet

pBIN = np.fromfile(fname, dtype=np.uint16)
# pBIN = np.fromfile(fname)
filename = os.path.basename(fname)
mdate = os.path.getmtime(fname)
logging.debug("uint16 file size: %d" % (pBIN.size))

if pBIN[0] != PKT_SZ:
    logging.error("Invalid frame size, should be: 0x%x." % (PKT_SZ))
    
logging.debug("First word: 0x%x." % (pBIN[0]))

step = (PKT_SZ+4)//2  # steps in 16-bit words, add additional byte by streamer
logging.debug("step is (0x%x) %d 16-bit words" % (step,step))

pkt_szs = pBIN[::step]
findx = np.array_equal(pkt_szs, (np.full(pkt_szs.size, PKT_SZ, np.uint16)))
if not findx:
    logging.error('packet sizes are not equal')
else:
    logging.debug("Number of packets in file is %d" % (pkt_szs.shape[0]))
    
if (PKT_SZ+4) * pkt_szs.shape[0] != pBIN.size*2:
    logging.error('file contains partial frames')

# # skip headers and keep data only
pkt_idx = step*np.arange(pkt_szs.shape[0]) + HEADER_SZ
data_idx = np.arange(NSAMPLES) + pkt_idx[:, np.newaxis]
data = pBIN[data_idx]
# drop 1st 0x0 sample
data = data[:, 1:]
print(data)
logging.debug("Data shape is" + str(data.shape))

fig_id+=1
plt.figure(fig_id,figsize=(8,6),dpi=150)
# plt.legend(frameon=False)
plt.xlabel('Samples [0 to %d]' % (data.shape[0]-1))
plt.ylabel('Amplitude (ADU)')
plt.title('ePix HR Test Structure Samples')
# plt.plot(data[0,:])
# plt.plot(data[1,:])
# plt.plot(data[2,:])
#plt.plot(data[:,1::2])
plt.plot(data)

# dac_resol = 2**16-1
# dac_mem = 1024
# angle = 2*np.pi*np.linspace(0, 1, dac_mem)
# sine = -1*dac_resol/2*np.sin(angle)+dac_resol/2

# plt.plot(np.tile(sine,10),'r')


plt.show()

