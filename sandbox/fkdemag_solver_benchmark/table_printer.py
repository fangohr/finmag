"""
Contains helpers to build and output tables which look like the following.

        |   xlabel0     xlabel1      xlabel2 ...  xlabeln
--------|----------------------------------------------------
ylabel0 |    entry0      entry1       entry2 ...   entryn
ylabel1 |    entry0      entry1       entry2 ...   entryn
ylabel2 |    entry0      entry1       entry2 ...   entryn
...     |    entry0      entry1       entry2 ...   entryn
ylabeln |    entry0      entry1       entry2 ...   entryn

Note that the output is performed one entry at a time, one row after the other.
This means that extraneous output will break the table.

The advantage is that this can be used for continually showing the progress
of some long-running operation (the result of which depends on the two variables
described by the labels) in a way which is informative and nice to look at.

"""


def table_header(xlabels, xlabels_width_min=10, ylabels_width=15):
    """
    Output the table header.

    Given the list of strings `xlabels`, this will print the table header
    using the strings as labels and return the width of the columns
    associated with them.

    The xlabels will be right-aligned and short xlabels will be padded to
    a minimum width of 10. That value can be changed using `xlabels_width_min`.

    The first column width is those of the ylabels column, which can be
    controlled with the optional argument `ylabels_width` (default=15).

    """
    columns = [ylabels_width]
    print " " * columns[0] + "|",
    for label in xlabels:
        width = max(xlabels_width_min, len(label))
        print "{:>{w}}".format(label, w=width),
        columns.append(width)
    print "\n" + "-" * columns[0] + "|" + "-" * (len(columns) + sum(columns[1:])),
    return columns


def row_start(ylabel, width):
    """
    Start a new table row with the label `ylabel` of width `width`.
    The label will be right-aligned to fit `width`.

    """
    print "\n{:<{w}}|".format(ylabel, w=width),


def row_entry(entry, width, fmt="{:>{w}.3}"):
    """
    Print the table entry `entry` of width `width`.

    By default it will use a format for floating point numbers, which it will
    right-align with spaces to fill `width`. This behaviour can be controlled
    py passing a new format string in `fmt` (with enclosing curly braces). The
    width is passed to that format string as `w` and must be used or else
    python will raise a KeyError.

    """
    print fmt.format(entry, w=width),


def test_usage_example():
    xlabels = ["True", "False"]
    ylabels = ["True", "False"]

    print "Conjunction.\n"
    widths = table_header(xlabels, xlabels_width_min=6, ylabels_width=6)
    for i, bv1 in enumerate(ylabels):
        row_start(bv1, widths[0])
        for j, bv2 in enumerate(xlabels):
            row_entry(bv1 and bv2, widths[j + 1], "{:>{w}}")

if __name__ == "__main__":
    test_usage_example()
