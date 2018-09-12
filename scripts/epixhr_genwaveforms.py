#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys

# constants
dac_resol = 2**16-1
dac_mem = 1024
shrinkfactor = 0.40 / 2
shrink = shrinkfactor * dac_resol  # shrink 15% from top and %15 from bottom = 30%
# shrink = 0
# shrinkfactor = 0


# incr ramp
ramp = np.linspace(0+shrink, dac_resol-shrink, dac_mem, dtype=int)
np.savetxt("ramp.csv", ramp, fmt="%d")

# decr ramp
rampf = np.linspace(dac_resol-shrink, 0+shrink, dac_mem, dtype=int)
np.savetxt("rampf.csv", rampf, fmt="%d")

# sine
angle = 2*np.pi*np.linspace(0, 1, dac_mem)
offset = dac_resol/2
amp = (1-2*shrinkfactor)*dac_resol/2  # 70% of full scale
sine = amp*np.sin(angle)+offset
np.savetxt("sine.csv", sine, fmt="%d")

if len(sys.argv) > 1 and sys.argv[1] == '-p':
    plt.figure(1,figsize=(8,6),dpi=150)
    plt.xlabel('Time (Bin)')
    plt.ylabel('Amplitude (ADU)')
    plt.title('DAC Waveforms')
    plt.plot(ramp)
    plt.plot(rampf)
    plt.plot(sine)
    plt.show()
