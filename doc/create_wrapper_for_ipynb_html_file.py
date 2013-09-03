#!/usr/bin/env python

# This script takes a .ipynb file as input and generates a minimal
# .rst file that wraps the converted .html output of the ipython
# notebook using sphinx's :raw: directive. It also adds a title, which
# is extracted from the notebook cells if present (otherwise it just
# uses the input filename as title).

import re
import os
import sys
import json
import textwrap
import itertools

try:
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
except IndexError:
    print "Usage: {} INPUT.ipynb OUTPUT.rst".format(__file__)
    sys.exit()


def get_level(cell):
    try:
        level = cell['level']
    except KeyError:
        # Return the number of leading '#' as the level
        m = re.search('^(#+)', cell['source'][0])
        level = len(m.group(1))
    return level


def extract_title(cell):
    if cell['cell_type'] == 'heading':
        title = cell['source'][0]
        print "[DDD] HEADING: Using cell title: {}".format(title)
    elif cell['cell_type'] == 'markdown':
        # Remove leading '#' signs
        title = re.sub('^#+\s*', '', cell['source'][0])
        print "[DDD] MARKDOWN: Using cell title: {}".format(title)
    else:
        raise RuntimeError("Unknown cell type: {}".format(cell['cell_type']))
    return title


# Read JSON data
json_data = open(infilename).read()
data = json.loads(json_data)

# Extract all cells from all worksheets
cells = list(itertools.chain(*[ws['cells'] for ws in data['worksheets']]))

header_cells = sorted([c for c in cells if (c['cell_type'] == 'heading') or (c['cell_type'] == 'markdown' and c['source'][0].startswith('#'))], key=get_level)

if header_cells == []:
    title = os.path.basename(infilename)
    print "No header cells found. Using filename as title: '{}'".format(title)
else:
    title = extract_title(header_cells[0])
    print "Found some header cells. Using title from first highest-level one: '{}'".format(title)

with open(outfilename, 'w') as f:
    outbasename, _ = os.path.splitext(os.path.basename(outfilename))
    f.write(textwrap.dedent("""\
        {}
        {}

        .. raw:: html
           :file: {}
        """.format(title, '=' * len(title), outbasename + '.html')))
