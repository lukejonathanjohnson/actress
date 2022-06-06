#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:12:24 2020

@author: lukejohnson1
"""


#
#  This file is part of actress.
#
#  actress is software for use only by permission at this time;
#  A version for public use will follow at a later date.
#
"""
actress is a package to simulate stellar variability due to spots and facular
regions on magnetically active late-type stars
"""

# from .handyfuncs import (
#     FFS14, QfromS, S2Lat, CalcVar, SFF14, Qfromff, loadnpy, distlims, rackrf,
#     HEMtoR, lonband, Qradrat, MSHtoR, makeffgrid, find_nearest
#     )


try:
    from .version import __version__
except:
    try:
        from actress.version import __version__
    except:
        from version import __version__

try:
    from .handyfuncs import *
except:
    try:
        from actress.handyfuncs import *
    except:
        from handyfuncs import *

try:
    from .hmap import *
except:
    try:
        from actress.hmap import *
    except:
        from hmap import *

try:
    from .simulator import *
except:
    try:
        from actress.simulator import *
    except:
        from simulator import *
