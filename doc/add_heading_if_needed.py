#!/usr/bin/env python

"""
Scan a given file 'FILENAME.rst' for header lines up to a certain
level. If no such lines are encountered, a header line of the
following form is added at the top of the file:

   FILENAME.ipynb
   ==============
"""

import sys
import re
import os

# Heading symbols used by nbconvert, in descending order (from
# "heading 1" to "heading 6"). Note that the dash needs to be escaped!
heading_symbols = ["=", "\-", "`", "'", ".", "~"]

# Check command line arguments
try:
    filename = sys.argv[1]
    if not filename.endswith('.rst'):
        raise ValueError("Filename must end in '.rst' "
                         "(got: '{}')".format(filename))
except IndexError:
    print "Error: filename expected as first argument."
    sys.exit(1)

try:
    max_allowed_level = int(sys.argv[2])
except IndexError:
    max_allowed_level = 1

print "Looking for header lines up to maximum level " + \
    "'{}' in file '{}'".format(max_allowed_level, filename)


# Create a regexp which we can use to filter the lines in the file
heading_symbol_str = ''.join(heading_symbols[0:max_allowed_level])
heading_regexp = '^[{}]+$'.format(heading_symbol_str)

# Filter lines that only contain a repetition of one of the allowed
# symbols. Note that we read all lines into memory. Since notebooks
# usually shouldn't be very large, this is hopefully ok.
lines = open(filename).readlines()
num_heading_lines = len(filter(None, [re.match(heading_regexp, l.rstrip())
                                      for l in lines]))

if num_heading_lines == 0:
    print "No header lines found (up to maximum level {}). Adding a header " \
          "line containing the filename at the beginning of the file.".format(
        max_allowed_level)

    # Write a title consisting of the filename at the beginning of the
    # file, then dump the lines back that we read in earlier.
    with open(filename, 'w') as f:
        title_name = os.path.basename(filename)
        f.write(title_name + '\n')
        f.write('=' * len(title_name) + '\n\n')
        for l in lines:
            f.write(l)
else:
    print "Found {} header line(s); not altering file.".format(num_heading_lines)
