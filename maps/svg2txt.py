"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Converts an svg map file to a text format.
# Args:
# fname - path to .svg file
# numXTransm, numYTransm - beacons are placed on an evenly spaced grid of numXTransm x numYTransm locations.

from __future__ import print_function
import re
import sys

def svg2txt(fname,ntx,nty):

    f = open(fname,'rb').read()
    width = float(re.search(
        '\<svg[^\>]*width=[\"\']([\d\.]+)',f).group(1))
    height = float(re.search(
        '\<svg[^\>]*height=[\"\']([\d\.]+)',f).group(1))

    paths = re.findall('\<path[^\>]*\sd=\"([^\"]*)',f)

    walls = list()
    for p in paths:
        # Tokenize and parse p
        prev = None
        mode = 0
        cur = [0,0]
        d = p

        while len(d) > 0:
            if d[0] == 'M' or d[0] == 'L':
                mode = 1
                d = d[2:]
            elif d[0] == 'm':
                mode = 0
                d = d[2:]
            else:
                q = re.match('([\d\.\-e]*),([\d\.\-e]*)\s*(.*)',d)
                x=float(q.group(1))
                y=float(q.group(2))
                d = q.group(3)

                if mode == 0:
                    x = x + cur[0]
                    y = y + cur[1]

                if prev is not None:
                    walls.append(tuple([prev[0],prev[1],x,y]))

                prev = [x,y]
                cur = [x,y]

    print("%.2f %.2f" % (width,height))
    print("%d %.d" % (ntx,nty))
    for j in range(len(walls)):
        print('%.2f %.2f %.2f %.2f' % walls[j])


if len(sys.argv) != 4:
    sys.exit('USAGE: svg2txt file.svg numXTransm numYTransm')

svg2txt(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
