#!/usr/bin/env python
"""
simple example script for running and testing notebooks.

Usage: `ipnbdoctest.py foo.ipynb [bar.ipynb [...]]`

Each cell is submitted to the kernel, and the outputs are compared
with those stored in the notebook.
"""

import os
import sys
import time
import base64
import re

from collections import defaultdict
from Queue import Empty

from IPython.zmq.blockingkernelmanager import BlockingKernelManager
from IPython.nbformat.current import reads, NotebookNode


class IPythonNotebookDoctestError(Exception):
    pass


def compare_png(a64, b64):
    """compare two b64 PNGs (incomplete)"""
    try:
        import Image
    except ImportError:
        pass
    adata = base64.decodestring(a64)
    bdata = base64.decodestring(b64)
    return True


def sanitize(s):
    """
    Sanitize a string for comparison.

    Fix universal newlines, strip trailing newlines, and normalize
    likely random values (memory addresses and UUIDs).
    """
    # normalize newline:
    s = s.replace('\r\n', '\n')

    # ignore trailing newlines (but not space)
    s = s.rstrip('\n')

    # normalize hex addresses:
    s = re.sub(r'0x[a-f0-9]+', '0xFFFFFFFF', s)

    # normalize UUIDs:
    s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)

    # remove timestamps in Finmag logging output
    s = re.sub(r'\[201\d-\d\d-\d\d \d\d:\d\d:\d\d\]', 'LOGGING_TIMESTAMP', s)

    # ignore version information of external dependencies
    s = re.sub('^paraview version .*$', 'PARAVIEW_VERSION', s)
    for dep in ['Finmag', 'Dolfin', 'Matplotlib', 'Numpy', 'Scipy', 'IPython',
                'Python', 'Paraview', 'Sundials', 'Boost-Python', 'Linux']:
        s = re.sub('DEBUG: %20s: .*' % dep, 'VERSION_%s' % dep, s)

    # ignore specific location of logging output file
    s = re.sub("Finmag logging output will be.*", "FINMAG_LOGGING_OUTPUT", s)

    # ignore datetime objects
    s = re.sub(r'datetime.datetime\([0-9, ]*\)', 'DATETIME_OBJECT', s)

    return s


def consolidate_outputs(outputs):
    """consolidate outputs into a summary dict (incomplete)"""
    data = defaultdict(list)
    data['stdout'] = ''
    data['stderr'] = ''

    for out in outputs:
        if out.type == 'stream':
            data[out.stream] += out.text
        elif out.type == 'pyerr':
            data['pyerr'] = dict(ename=out.ename, evalue=out.evalue)
        else:
            for key in ('png', 'svg', 'latex', 'html',
                        'javascript', 'text', 'jpeg'):
                if key in out:
                    data[key].append(out[key])
    return data


def compare_outputs(test, ref, skip_compare=('png', 'traceback',
                                             'latex', 'prompt_number')):
    for key in ref:
        if key not in test:
            print "missing key: %s != %s" % (test.keys(), ref.keys())
            return False
        elif (key not in skip_compare) and \
                (sanitize(test[key]) != sanitize(ref[key])):
            print "----- Mismatch {}: --------------------------------------------------------".format(key)
            print test[key]
            print "----------   !=   ----------"
            print ref[key]
            print "--------------------------------------------------------------------------------"
            return False
    return True


def run_cell(km, cell):
    shell = km.shell_channel
    iopub = km.sub_channel
    # print "\n\ntesting:"
    # print cell.input
    shell.execute(cell.input)
    # wait for finish, maximum 20s
    shell.get_msg(timeout=20)
    outs = []

    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue

        content = msg['content']
        # print msg_type, content
        out = NotebookNode(output_type=msg_type)

        if msg_type == 'stream':
            out.stream = content['name']
            out.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            for mime, data in content['data'].iteritems():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'pyout':
                out.prompt_number = content['execution_count']
        elif msg_type == 'pyerr':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        else:
            print "unhandled iopub msg:", msg_type

        outs.append(out)
    return outs


def test_notebook(nb):
    km = BlockingKernelManager()
    km.start_kernel(extra_arguments=['--pylab=inline'],
                    stderr=open(os.devnull, 'w'))
    km.start_channels()
    # run %pylab inline, because some notebooks assume this
    # even though they shouldn't
    km.shell_channel.execute("pass")
    km.shell_channel.get_msg()
    while True:
        try:
            km.sub_channel.get_msg(timeout=1)
        except Empty:
            break

    successes = 0
    failures = 0
    errors = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            try:
                outs = run_cell(km, cell)
            except Exception as e:
                print "failed to run cell:", repr(e)
                print cell.input
                errors += 1
                continue

            failed = False
            for out, ref in zip(outs, cell.outputs):
                if not compare_outputs(out, ref):
                    failed = True
            if failed:
                failures += 1
            else:
                successes += 1
            sys.stdout.write('.')

    print
    print "tested notebook %s" % nb.metadata.name
    print "    %3i cells successfully replicated" % successes
    if failures:
        print "    %3i cells mismatched output" % failures
    if errors:
        print "    %3i cells failed to complete" % errors
    km.shutdown_kernel()
    del km
    if failures or errors:
        raise IPythonNotebookDoctestError(
            "The notebook {} failed to replicate successfully.".format(
                nb.metadata['name']))


if __name__ == '__main__':
    for ipynb in sys.argv[1:]:
        print "testing %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        test_notebook(nb)
