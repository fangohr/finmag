#!/usr/bin/env python

# IPython notebook testing script. This is based on the gist [1] (revision 5).
#
# Each cell is submitted to the kernel, and the outputs are compared
# with those stored in the notebook. If the output is an image this is
# currently ignored.
#
# https://gist.github.com/minrk/2620735


"""
IPython notebook testing script.

Usage: `ipnbdoctest.py foo.ipynb [bar.ipynb [...]]`

"""

import os
import re
import sys
import time
import base64
import string

from collections import defaultdict
from Queue import Empty

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager \
        import BlockingKernelManager as KernelManager

from IPython.nbformat.current import reads, NotebookNode


CELL_EXECUTION_TIMEOUT = 200  # abort cell execution after this time (seconds)

# If any of the following patterns occurs in the output, the line
# containing it will be discarded. If you don't want to discard the
# entire line but only bits of it, don't use "discard_patterns" but
# rather add an explicit replacement rule in the function 'sanitize()'
# below.
#
# On the other hand, if a line may only occur in the output under
# certain circumstances (such as the matplotlib user warning), it must
# be added here instead of in sanitize(), because otherwise there may
# still be an empty line in the reference output which is not matched
# in the computed output, or vice versa.
DISCARD_PATTERNS = ["Warning: Ignoring netgen's output status of 34304",
                    "UserWarning: This figure includes Axes that are not compatible with tight_layout, so its results might be incorrect",
                    "DEBUG: Found unused display :[0-9]+",
                    "DEBUG: Rendering Paraview scene on display :[0-9]+ using xpra.",
                    ]


# This is used to remove coloring from error strings (which makes them unreadable in Jenkins)
ANSI_COLOR_REGEX = "\x1b\[(\d+)?(;\d+)*;?m"
def decolorize(string):
    return re.sub(ANSI_COLOR_REGEX, "", string)


def indent(s, numSpaces):
    """
    Indent all lines except the first one with numSpaces spaces.
    """
    s = string.split(s, '\n')
    s = s[:1] + [(numSpaces * ' ') + line for line in s[1:]]
    s = string.join(s, '\n')
    return s


class IPythonNotebookDoctestError(Exception):
    pass


def compare_png(a64, b64):
    """
    Compare two b64 PNGs (incomplete).

    """
    try:
        import Image
    except ImportError:
        pass
    adata = base64.decodestring(a64)
    bdata = base64.decodestring(b64)
    return True


def matches_some_discard_pattern(s):
    """
    Helper function to check whether any of the patterns in
    DISCARD_PATTERNS occurs in the string.

    """
    for pat in DISCARD_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def keep_cell_output(out):
    return not (out['output_type'] == 'stream' and matches_some_discard_pattern(out['text']))


def sanitize(s):
    """
    Sanitize a string for comparison.

    Fix universal newlines, strip trailing newlines, and normalize
    likely random values (date stamps, memory addresses, UUIDs, etc.).
    """
    # normalize newline:
    s = s.replace('\r\n', '\n')

    # ignore trailing newlines (but not space)
    s = s.rstrip('\n')

    # normalize hex addresses:
    s = re.sub(r'0x[a-f0-9]+', '0xFFFFFFFF', s)

    # normalize UUIDs:
    s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)

    ##
    ## Finmag-related stuff follows below
    ##

    # Remove timestamps in logging output
    s = re.sub(r'\[201\d-\d\d-\d\d \d\d:\d\d:\d\d\]', 'LOGGING_TIMESTAMP', s)

    # Ignore version information of external dependencies
    s = re.sub('^paraview version .*$', 'PARAVIEW_VERSION', s)
    for dep in ['Finmag', 'Dolfin', 'Matplotlib', 'Numpy', 'Scipy', 'IPython',
                'Python', 'Paraview', 'Sundials', 'Boost-Python', 'Linux']:
        s = re.sub('DEBUG: %20s: .*' % dep, 'VERSION_%s' % dep, s)

    # Ignore specific location of logging output file
    s = re.sub("Finmag logging output will be.*", "FINMAG_LOGGING_OUTPUT", s)

    # Ignore datetime objects
    s = re.sub(r'datetime.datetime\([0-9, ]*\)', 'DATETIME_OBJECT', s)

    # Warning coming from Matplotlib occasionally. The warning comes
    # from a different line in different versions of matplotlib and
    # thus results in a failed comparison. Replace with empty string.
    s = re.sub(r'.*UserWarning: This figure includes Axes that are not '
               'compatible with tight_layout, so its results might be '
               'incorrect.*', '', s)

    # If a mesh exists already, we get a different message from
    # generation of the mesh.
    s = re.sub(r'.*The mesh.*already exists and is automatically '
                'returned.', '', s)


# def consolidate_outputs(outputs):
#     """consolidate outputs into a summary dict (incomplete)"""
#     data = defaultdict(list)
#     data['stdout'] = ''
#     data['stderr'] = ''
#
#     for out in outputs:
#         if out.type == 'stream':
#             data[out.stream] += out.text
#         elif out.type == 'pyerr':
#             data['pyerr'] = dict(ename=out.ename, evalue=out.evalue)
#         else:
#             for key in ('png', 'svg', 'latex', 'html',
#                         'javascript', 'text', 'jpeg'):
#                 if key in out:
#                     data[key].append(out[key])
#     return data


def report_mismatch(key, test, ref, cell, message):
    """
    See compare_outputs - this is just a helper function
    to avoid re-writing code twice.
    """

    output = "\n"
    output += "{}\n".format(message)
    output += "We were processing the following cell.input:\n"
    output += "--- Cell input start\n"
    output += "{}\n".format(cell.input)
    output += "--- Cell input end\n"

    if key in test and key in ref:
        output += "----- Mismatch for key='{}': ---------------------------------------------------\n".format(key)
        output += "{}\n".format(test[key])
        output += "----------   !=   ----------"
        output += "{}\n".format(ref[key])
        output += "--------------------------------------------------------------------------------\n"
    else:
        output += "Failure with key='{}'\n".format(key)
        if not key in test:
            output += "key not in test output\n"
        if not key in ref:
            output += "key not in ref output\n"
            assert False, "I think this should be impossible - is it? HF, Dec 2013"

    output += "--- Test output, with keys -> values:\n"
    for k, v in test.items():
        if k == 'traceback':
            v = indent('\n'.join(map(decolorize, v)), 41)
        output += "\tTest output:       {:10} -> {}\n".format(k, v)
    output += "--- Reference output, with keys -> values:\n"
    for k, v in ref.items():
        if k == 'png':
            v = '<PNG IMAGE>'
        output += "\tReference output:  {:10} -> {}\n".format(k, v)
    output += "- " * 35 + "\n"
    return output


def compare_outputs(test, ref, cell, skip_compare=('png', 'traceback',
                                                   'latex', 'prompt_number',
                                                   'metadata')):
    """
    **Parameters**

     ``test`` is the output we got from executing the cell

     ``ref`` is the reference output we expected from the saved
             notebook

     ``cell`` is the cell we are working on - useful to display the input
             in case of a fail

     ``skip_compare`` a list of cell types we ignore
    """

    for key in ref:
        if key not in test:
            # Note: One possibility for this branch is if an exception
            # is raised in the notebook. Typical keys in the test notebook
            # are ['evalue', 'traceback', 'ename']. Let's report some more
            # detail in this case.
            output = report_mismatch(key, test, ref, cell, "Something went wrong")
            print(output)
            # Now we have printed the failure. We should create a nice
            # html snippet, the following is a hack to get going quickly.
            # (HF, Dec 2013)
            htmlSnippet = "Not HTML, just tracking error:<br><br>\n\n" + output
            return False, htmlSnippet
        elif (key not in skip_compare) and \
                (sanitize(test[key]) != sanitize(ref[key])):
            output = report_mismatch(key, test, ref, cell, "In more detail:")
            print(output)
            try:
                import diff_match_patch
                dmp = diff_match_patch.diff_match_patch()
                diffs = dmp.diff_main(sanitize(ref[key]), sanitize(test[key]))
                dmp.diff_cleanupSemantic(diffs)
                htmlSnippet = dmp.diff_prettyHtml(diffs)
            except ImportError:
                print("The library 'diff-match-patch' is not installed, thus "
                      "no diagnostic HTML output of the failed test could be "
                      "produced. Please consider installing it by saying "
                      "'sudo pip install diff-match-patch'")
            return False, htmlSnippet
    return True, ""


def run_cell(shell, iopub, cell):
    # print cell.input
    shell.execute(cell.input)
    # wait for finish, abort if timeout is reached
    shell.get_msg(timeout=CELL_EXECUTION_TIMEOUT)
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
            out['metadata'] = content['metadata']
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


def merge_streams(outputs):
    """
    Since the Finmag logger uses streams, output that logically
    belongs together may be split up in the notebook source. Thus we
    merge it here to be able to compare streamed output robustly.

    """
    outputs = filter(keep_cell_output, outputs)

    if outputs == []:
        return []

    res = outputs[:1]
    for out in outputs[1:]:
        prev_out = res[-1]
        if (prev_out['output_type'] == 'stream' and
            out['output_type'] == 'stream' and
            prev_out['stream'] == out['stream']):
            prev_out['text'] += out['text']
        else:
            res.append(out)

    return res


def test_notebook(nb):
    km = KernelManager()
    km.start_kernel(extra_arguments=['--pylab=inline'], stderr=open(os.devnull, 'w'))
    try:
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    except AttributeError:
        # IPython 0.13
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel

    # run %pylab inline, because some notebooks assume this
    # even though they shouldn't
    shell.execute("pass")
    shell.get_msg()
    while True:
        try:
            iopub.get_msg(timeout=1)
        except Empty:
            break

    successes = 0
    failures = 0
    errors = 0
    html_diffs_all = ""
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            try:
                # Ignore output from cells whose input starts
                # with the string '# IPYTHON_TEST_IGNORE_OUTPUT'.
                first_line = cell['input'].splitlines()[0] if (cell['input'] != '') else ''
                if re.search('^#\s*IPYTHON_TEST_IGNORE_OUTPUT', first_line):
                    outs = []
                else:
                    outs = run_cell(shell, iopub, cell)
            except Exception as e:
                print "failed to run cell:", repr(e)
                print cell.input
                errors += 1
                continue

            failed = False
            outs_merged = merge_streams(outs)
            cell_outputs_merged = merge_streams(cell.outputs)
            for out, ref in zip(outs_merged, cell_outputs_merged):
                cmp_result, html_diff = compare_outputs(out, ref, cell)
                html_diffs_all += html_diff
                if not cmp_result:
                    failed = True
            if failed:
                print "Failed to replicate cell with the following input: "
                print "=== BEGIN INPUT ==================================="
                print cell.input
                print "=== END INPUT ====================================="
                if failures == 0:
                    # This is the first cell that failed to replicate.
                    # Let's store its output for debugging.
                    first_failed_input = cell.input
                    first_failed_output = outs_merged
                    first_failed_output_expected = cell_outputs_merged
                    # For easier debugging, replace the (usually huge) binary
                    # data of any pngs appearing in the expected or computed
                    # output with a short string representing the image.
                    for node in first_failed_output_expected + first_failed_output:
                        try:
                            node['png'] = '<PNG IMAGE>'
                        except KeyError:
                            pass
                failures += 1
            else:
                successes += 1
            sys.stdout.write('.')
            sys.stdout.flush()

    if failures >= 1:
        outfilename = 'ipynbtest_failed_test_differences.html'
        with open(outfilename, 'w') as f:
            f.write(html_diffs_all)
        print("Diagnostic HTML output of the failed test has been "
              "written to '{}'".format(outfilename))

    print ""
    print "tested notebook %s" % nb.metadata.name
    print "    %3i cells successfully replicated" % successes
    if failures:
        print "    %3i cells mismatched output" % failures
    if errors:
        print "    %3i cells failed to complete" % errors
    kc.stop_channels()
    km.shutdown_kernel()
    del km
    if failures or errors:
        errmsg = ("The notebook {} failed to replicate successfully."
                  "".format(nb.metadata['name']))
        if failures:
            errmsg += \
                ("Input and output from first failed cell:\n"
                 "=== BEGIN INPUT ==================================\n"
                 "{}\n"
                 "=== BEGIN EXPECTED OUTPUT ========================\n"
                 "{}\n"
                 "=== BEGIN COMPUTED OUTPUT ========================\n"
                 "{}\n"
                 "==================================================\n"
                 "".format(first_failed_input,
                           first_failed_output_expected,
                           first_failed_output))
        raise IPythonNotebookDoctestError(errmsg)

if __name__ == '__main__':
    for ipynb in sys.argv[1:]:
        print "testing %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        test_notebook(nb)
        sys.stdout.flush()
