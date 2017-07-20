#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

pwd = os.getcwd()

pathlist = Path(pwd).glob('**/*.dat')
for path in pathlist:
    p, f = os.path.split(str(path))
    os.chdir(p)
    os.system("python3 ~/epix10ka/cac/cac/plot_epix10ka_stat.py " +
              "-i %s -m 0x3fff -b -e -v &> %s_out.txt" % (f, f))
    os.chdir(pwd)
