#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dump hdf5 combined epix10ka data for offline analysis.
dump includes
    (a) any available epix10ka cam
    (b) Beam Monitor
    (c) Acqiris
    (d) Wave8


:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20180105
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
import Detector.PyDataAccess as pda
import logging
from mpi4py import MPI
import numpy as np
import psana as pa
import sys
import time

""" Ways to run this script:
# Single local machine
mpirun -n 8 python dump_data_mpi.py

# batch system on psanaq
bsub -q psanaq -n 16 -o /reg/d/psdm/MFX/mfxx32516/scratch/abunimeh/%J.log \
  mpirun python dump_data_mpi.py -r RUN_NUMBER

# batch system when we are live in MFX
bsub -q psfehhiprioq -n 16 -o /reg/d/psdm/MFX/mfxx32516/scratch/abunimeh/%J.log \
  mpirun python dump_data_mpi.py -r RUN_NUMBER

# to span across multiple hosts add:
-R "span[ptile=1]"
"""

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", help="Show debugging info.")
parser.add_argument("-r", "--run", nargs=1, metavar=('RUN'), type=int, help="run number.",
                    required=True)

args = parser.parse_args()

if args.verbose:
    myloglevel = logging.DEBUG
else:
    myloglevel = logging.INFO

# set logging based on args
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=myloglevel)

EXPSTR = 'exp=mfxx32516:run=%d:smd' % (args.run[0])
# EXPSTR = 'exp=mfxx32516:run=%d:smd:dir=/reg/d/ffb/mfx/mfxx32516/xtc:live' % (args.run[0])

# 'MfxEndstation.0:Epix10ka.0'  # spaghetti
# 'MfxEndstation.0:Epix10ka.1'  # meatballs
# 'MfxEndstation.0:Epix10ka.2'  # cheese

# filename = "".join([c for c in EXPSTR if c.isalnum()]).rstrip() + '.h5'
filename = "run_%d.h5" % (args.run[0])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dsrc = pa.MPIDataSource(EXPSTR)
detNames = pa.DetNames()  # detector names
mdets = []  # list of detectors
mdets_sname = []  # list of short detector names
mdets_idx = []  # list of detector indicies

# only archive detectors of interest
for idx, detname in enumerate(detNames):
    sdet = detname[0].lower()
    if "epix10ka" in sdet or "beammon" in sdet or "acqiris" in sdet or "wave8" in sdet:
        mdets.append(detname[0])
        mdets_sname.append(detname[1])
        mdets_idx.append(idx)

# lists to keep track of detectors
mdets_d = []
mdets_s = []
mdets_epix10ka_idx = []
mdets_acqiris_idx = []
mdets_wave8_idx = []
mdets_beammon_idx = []
for idx, mdet in enumerate(mdets):
    mdets_d.append(pa.Detector(mdet))
    mdets_s.append(pa.Source(mdet))
    if "epix10ka" in mdet.lower():
        mdets_epix10ka_idx.append(idx)
    elif "acqiris" in mdet.lower():
        mdets_acqiris_idx.append(idx)
    elif "wave8" in mdet.lower():
        mdets_wave8_idx.append(idx)
    elif "beammon" in mdet.lower():
        mdets_beammon_idx.append(idx)

# no need to continue if there are no epix10ka
if len(mdets_epix10ka_idx) == 0:
    logging.error('no epix10ka cameras')
    sys.exit(0)

# gather_interval set 1 instead of 100 as recommended by cpo@
smldata = dsrc.small_data(filename, gather_interval=1)

if rank == 0:
    logging.info('looping thru events')
start = time.time()
for nevt, evt in enumerate(dsrc.events()):
    if nevt % 100 == 0:
        # print '***',rank,nevt,nevt/float(time.time()-start)*size
        logmsg = '***', rank, nevt, nevt / float(time.time() - start) * size
        logging.debug(logmsg)
        # pass

    # fresh dict
    d = {}

    epix10ka_data_valid = 0
    for i, epix10ka_idx in enumerate(mdets_epix10ka_idx):
        epix10ka_det = mdets_d[epix10ka_idx]
        raw = epix10ka_det.raw(evt)
        if raw is not None:
            epix10ka_data_valid = 1
            dataob = pda.get_epix_data_object(evt, mdets_s[epix10ka_idx])
            env = dataob.environmentalRows()
            calib = dataob.calibrationRows()
            epix10ka_data = {mdets_sname[epix10ka_idx]: {'calib': calib, 'raw': raw, 'env': env}}
            d.update(epix10ka_data)

    if epix10ka_data_valid == 1:
        for i, acqiris_idx in enumerate(mdets_acqiris_idx):
            acqiris_det = mdets_d[acqiris_idx]
            raw = acqiris_det.raw(evt)
            if raw is not None:
                waveforms = acqiris_det.waveform(evt)
                times = acqiris_det.wftime(evt)
                acqiris_data = {'acqiris_%d' % (i): {'waveforms': waveforms, 'times': times}}
                d.update(acqiris_data)

        for i, wave8_idx in enumerate(mdets_wave8_idx):
            wave8_det = mdets_d[wave8_idx]
            w8dlist = wave8_det.raw(evt)
            if w8dlist:
                w8arr = np.array(w8dlist)
                wave8_data = {'wave8_%d' % (i): {'data': w8arr}}
                d.update(wave8_data)

        for i, bmon_idx in enumerate(mdets_beammon_idx):
            bmon_det = mdets_d[bmon_idx]
            fluxData = bmon_det.get(evt)
            if fluxData is not None:
                flux = -1*fluxData.peakA()
                bmon_data = {'bmon_%d' % (i): {'flux': flux}}
                d.update(bmon_data)

        smldata.event(d)
        # if nevt > 3:
        #    break

# save HDF5
smldata.save()
