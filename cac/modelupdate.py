#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse, compare, and patch parameters in SPICE model files
:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20171025
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""

import argparse
import logging
import re
import sys


def main():
    """Routine to plot data. This uses argparse to get cli params.

    Example script using CAC library.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debugging info.")
    parser.add_argument("-i", "--inputfiles", nargs=2, metavar=('SRC', 'DST'),
                        help="Parameters inserted from source to destination.", required=True)
    args = parser.parse_args()

    if args.verbose:
        myloglevel = logging.DEBUG
    else:
        myloglevel = logging.INFO

    # set logging based on args
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=myloglevel)

    sfile = args.inputfiles[0]
    dfile = args.inputfiles[1]

    # openfiles
    try:
        srcfile = open(sfile, 'r')
    except IOError as e:
        logging.error("[%s] cannot be opened!", sfile)
        logging.error(e)
        sys.exit(1)
    logging.debug("[%s] opened for reading", sfile)

    try:
        dstfile = open(dfile, 'r')
    except IOError as e:
        logging.error("[%s] cannot be opened!", dfile)
        logging.error(e)
        sys.exit(1)
    logging.debug("[%s] opened for reading", dfile)

    # create new file for merging
    try:
        newfile = open(dfile+'.new', 'w')
    except IOError as e:
        logging.error("[%s] cannot be opened!", dfile+'.new')
        logging.error(e)
        sys.exit(1)
    logging.debug("[%s] opened for writing", dfile+'.new')

    # read file content and store it as a list
    srcfile_content = srcfile.read().splitlines()

    found_model = 0  # flag to force a single loop
    params = dict()  # dictionary to store all parameters and values
    for i, line in enumerate(srcfile_content):
        # skip empty line
        if (len(line.strip())) == 0 or line[0:1] == "*":
            continue

        if (line[0:6] == ".model"):
            if not found_model:
                logging.debug("[%d] found model %s", i, line)
                found_model = 1
                continue
            else:
                logging.debug("[%d] found another model %s. Completed.", i, line)
                break
        if found_model:
            if (line[0:1] == "+"):
                line = line[1:]  # get rid of +
            line = line.rstrip()  # remove trailing spaces on right
            line = line.lstrip()  # remove spaces on left
            line = re.sub('\s\s+', '|', line)  # separate params with |
            # skip empty lines
            if len(line) == 0:
                continue
            # check if multiple params are on line
            if '|' in line:
                temp_param = dict((p.strip(), v.strip())
                                  for p, v in (it.split('=') for it in line.split('|')))
            else:  # single param
                single = [it.strip() for it in line.split('=')]
                temp_param = {single[0]: single[1]}
            # append param to dict
            params.update(temp_param)

    # clean up parameters
    upparams = dict()
    for p, v in params.items():
        if "'" in v and ('-' in v or '+' in v):
            v = v.strip("'")  # get id of quote
            # we are only interested in strnig after + or - sign
            if '+' in v:
                v = re.split('\+', v)[-1]
                v = '+' + v
            if '-' in v:
                v = re.split('-', v)[-1]
                v = '-' + v
            # store dict of values to be updated
            upparams.update({p: v})

    # destination
    dstfile_content = dstfile.read().splitlines()  # store lines in var
    for i, line in enumerate(dstfile_content):
        # skip empty line
        if (len(line.strip())) == 0 or line[0:1] == "*":
            continue

        for p, v in upparams.items():
            # look for parameter followed by a scientific notation number
            lookfor = r'\b'+p+'\s=\s-?\d[^\s]*'
            found = re.search(lookfor, line, re.IGNORECASE)
            if found:
                logging.debug('found [%d %d] ' % (found.start(), found.end()) + p)
                logging.debug('\t' + line + ' add ' + v + ' becomes')
                # inject value inline
                newline = line[:found.start() + len(p) + 3] + \
                    "'" + line[found.start() + len(p) + 3:found.end()] + \
                    v + "'" + line[found.end():]
                logging.debug('\t' + newline)
                dstfile_content[i] = newline  # update original var
                line = newline

    # prepare carriage returns and write them to new file.
    newfile.write("\n".join(dstfile_content))
    logging.debug(upparams)

    srcfile.close()
    dstfile.close()
    newfile.close()
    logging.debug("closing files")
    logging.info("Results stored in " + dfile + ".new")


if __name__ == "__main__":
    main()
