#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live alarms for humidity. Shows environmental data.

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20180105
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import Detector.PyDataAccess as pda
import numpy as np
import psana as pa
from psmon.plots import Image as ii
from psmon.plots import XYPlot
from psmon import publish
import sys
from PIL import Image, ImageFont, ImageDraw

DETECTOR0 = 'MfxEndstation.0:Epix10ka.1'
# DETECTOR1 = 'MfxEndstation.0:Epix10ka.1'

# average over NUMEVT
NUMEVT = 120

ds = pa.DataSource('shmem=psana.0:stop=no')
# ds = pa.DataSource('exp=mfxx32516:run=253:smd')

detNames = pa.DetNames()
for detname in detNames:
    print(detname)

src = pa.Source(DETECTOR0)
det = pa.Detector(DETECTOR0)
env = ds.env()

config = env.configStore().get(pa.Epix.Config10kaV1, src)
if not config:
    print('no config')
else:
    print('epix10ka is configed')

BTEMP_color = "green"
ATEMP_color = "green"
HU_color = "green"
AAi_color = "green"
ADi_color = "green"
DET_color = "green"
AV_color = "green"
DV_color = "green"

i = 0
j = 0
envs_s = np.zeros([3, NUMEVT], dtype=np.float32)  # signed
envs_u = np.zeros((5, NUMEVT), dtype=np.uint32)  # unsigned
for nevent, evt in enumerate(ds.events()):
    raw = det.raw(evt)
    if raw is not None:
        dataob = pda.get_epix_data_object(evt, src)
        envrows = dataob.environmentalRows()

        envs_s[0, i] = envrows[1, 0].astype(np.int32) / 100.0  # sbcs_temp, -19
        envs_s[1, i] = envrows[1, 1].astype(np.int32) / 100.0  # amb_temp, 6
        envs_s[2, i] = envrows[1, 2].astype(np.int32) / 100.0  # humidity, 4

        envs_u[0, i] = envrows[1, 3].astype(np.uint32)  # asic_ana_i, 1325.7
        envs_u[1, i] = envrows[1, 4].astype(np.uint32)  # asic_dig_i, 17.88
        envs_u[2, i] = envrows[1, 5].astype(np.uint32)  # det_gring_i, only for spaghetti 38.89
        envs_u[3, i] = envrows[1, 7].astype(np.uint32)  # ana_v, 5969.4
        envs_u[4, i] = envrows[1, 8].astype(np.uint32)  # dig_v, 6041.25

        i += 1

        # print(envs_s[:,i])
        # print(envs_u[:,i])

        if i % NUMEVT == 0:
            print('Frames Collected: %d\r' % (j)),
            sys.stdout.flush()
            j += 1

            envs_s_avg = envs_s.mean(axis=1)
            envs_u_avg = envs_u.mean(axis=1)

            i = 0
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 24)

            im = Image.new('RGB', (800, 200), (255, 0, 0))
            dr = ImageDraw.Draw(im)

            # dew point
            relative_humi = envs_s_avg[2]
            atmp_temp = envs_s_avg[1]
            t_d = (relative_humi / 100.0)**(1.0 / 8.0) * \
                (112.0 + 0.9 * atmp_temp) + 0.1 * atmp_temp - 112.0
            t_d += 5.0  # add 5 celsius as a safety margin
            # print(t_d)
            BTEMP = "BTEMP %.2f" % (envs_s_avg[0])
            # if dew point temp is larger than sbtemp then alarm
            if t_d > envs_s_avg[0]:
                BTEMP_color = "red"
            ATEMP = "ATEMP %.2f" % (envs_s_avg[1])
            HU = "HU %.2f" % (envs_s_avg[2])
            AAi = "AAi %.2f" % (envs_u_avg[0])
            ADi = "ADi %.2f" % (envs_u_avg[1])
            DET = "DET %.2f" % (envs_u_avg[2])
            AV = "AV %.2f" % (envs_u_avg[3])
            DV = "DV %.2f" % (envs_u_avg[4])

            dr.rectangle(((0, 0), (200, 100)), fill=BTEMP_color, outline="white")  # BTEMP
            dr.rectangle(((0, 100), (200, 200)), fill=ADi_color, outline="white")  # ADi
            dr.rectangle(((200, 0), (400, 100)), fill=ATEMP_color, outline="white")  # ATEMP
            dr.rectangle(((200, 100), (400, 200)), fill=DET_color, outline="white")  # DET
            dr.rectangle(((400, 0), (600, 100)), fill=HU_color, outline="white")  # HU
            dr.rectangle(((400, 100), (600, 200)), fill=AV_color, outline="white")  # AV
            dr.rectangle(((600, 0), (800, 100)), fill=AAi_color, outline="white")  # AAi
            dr.rectangle(((600, 100), (800, 200)), fill=DV_color, outline="white")  # DV

            dr.text((20, 50), BTEMP, font=font)
            dr.text((220, 50), ATEMP, font=font)
            dr.text((420, 50), HU, font=font)
            dr.text((620, 50), AAi, font=font)

            dr.text((20, 150), ADi, font=font)
            dr.text((220, 150), DET, font=font)
            dr.text((420, 150), AV, font=font)
            dr.text((620, 150), DV, font=font)
            npimg = np.array(im)
            npimg = npimg[:, :, 1]
            # print(npimg.shape)
            img0 = ii(0, "epix10ka_env", npimg)
            publish.send('IMAGE0', img0)
            detplotxy = XYPlot(0, "dr", range(NUMEVT), envs_u[2, :])
            publish.send('dr', detplotxy)
            huplotxy = XYPlot(0, "hu", range(NUMEVT), envs_s[2, :])
            publish.send('hu', huplotxy)
            sbplotxy = XYPlot(0, "sb", range(NUMEVT), envs_s[0, :])
            publish.send('sb', sbplotxy)
            # raw_input('Hit <CR> for next event')

        # raw_input('Hit <CR> for next event')
        # if nevent >= 2:
        #    break
