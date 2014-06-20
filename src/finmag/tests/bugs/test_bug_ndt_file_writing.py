import pytest

# coding: utf-8

# ## Saving spatially averaged magnetisation into a file

# Bug 8 June 2014. On some machines, we seem to have wrong data in the
# ndt file. Some inspection shows that there are too many columns in
# the ndt file.

# We didn't have a test for this, it was only caught by chance (Max
# looking at a plot in a notebook plotting the curves). Max writes:
#
#     Quick update for tonight: fortunately this doesn't seem to be
#     very serious. Increasing the number of points to be plotted
#     shows that it is always only the very first data point that is
#     off. So it seems that this is an issue with .ndt saving where
#     the first value is somehow not correctly updated.
#
#     I just did a bisection of the history (version control and
#     bisect FTW! :)) and the commit which introduced this buglet is
#     30145c2f9595 ("ndt file now saves max dmdt norm"). However, I
#     can't see offhand why that changeset would introduce the
#     bug. Anyway, off to sleep now. Just thought I'd send this
#     update. Once someone finds the cause of the bug we should also
#     add a regression test (it's a bit strange that this hasn't been
#     picked up by our existing tests).
#
#

# Note that there is an additional Ipython notebook file for this bug
# that is convenient for experimentation.


def test_ndt_writing_pretest():

    import finmag

    # In[4]:

    sim = finmag.example.barmini(name='bug-saving-average-data-june-2014')


    # What is the current magnetisation? We expect it to be $ \vec{m}
    # = [\sqrt(2), 0, \sqrt(2)]$ as this is the initial value in the
    # barmini example.

    # In[5]:

    import math
    m = sim.get_field_as_dolfin_function('m')
    points = [[0, 0, 0], [1, 0, 0], [2, 0,0 ], [0, 0, 5], [1, 1, 2],
              [3, 3, 10]]
    for point in points:
        print("m({}) = {}".format(point, m(point)))
        assert (m(point)[0] - math.sqrt(2)) < 1e-15
        assert (m(point)[1] - 0) < 1e-15
        assert (m(point)[2] - math.sqrt(2)) < 1e-15

    # now we know for sure what data we are writing.


def number_of_columns_in_ndt_file_consistent(ndtfile):
    lines = open(ndtfile).readlines()
    headers = lines[0][1:].split()  # string of leading hash in first line
    n = len(headers)
    print("Found {} headers: = {}".format(n, headers))
    # skip line 1 which contains the units
    for i in range(2, len(lines)):
        print("Found {} columns in line {}.".format(len(lines[i].split()), i))
    print("Printed the length to show all the data, now we test each line")
    for i in range(2, len(lines)):
        assert(len(lines[i].split()) == n)


def test_ndt_writing_correct_number_of_columns_1line():

    # Here we write only one line to the ndt file

    import finmag

    sim = finmag.example.barmini(name='bug-saving-average-data-june-2014-a')
    sim.save_averages()


    # The first line contains the title for every column, the second
    # line the (SI) units in which the entity is measured, and the
    # third and any other lines contain the actual data.

    # Check that all lines in this data file have the right number of
    # entries (columns)

    number_of_columns_in_ndt_file_consistent('bug_saving_average_data_june_2014_a.ndt')


#@pytest.mark.xfail

def test_ndt_writing_correct_number_of_columns_2_and_more_lines():

    # Here we write multiple lines to the ndt file

    import finmag

    # In[4]:

    sim = finmag.example.barmini(name='bug-saving-average-data-june-2014-b')

    sim.save_averages()

    # and write some more data
    sim.schedule("save_ndt", every=10e-12)
    sim.run_until(0.1e-9)

    # The first line contains the title for every column, the second
    # line the (SI) units in which the entity is measured, and the
    # third and any other lines contain the actual data.

    # Check that all lines in this data file have the right number of
    # entries (columns)

    number_of_columns_in_ndt_file_consistent(
        'bug_saving_average_data_june_2014_b.ndt')



def check_magnetisation_is_of_sensible_magnitude(ndtfile):
    import finmag
    data = finmag.util.fileio.Tablereader(ndtfile)
    mx, my, mz = data['m_x', 'm_y', 'm_z']
    print("Found {} saved steps.".format(len(mx)))
    assert len(mx) == len(my) == len(mz)
    # magnetisation should be normalised, so cannot exceed 1
    for m_x, m_y, m_z in zip(mx, my, mz):
        assert abs(m_x) <= 1, "m_x = {}".format(m_x)
        assert abs(m_y) <= 1, "m_x = {}".format(m_x)
        assert abs(m_z) <= 1, "m_x = {}".format(m_x)


def test_ndt_writing_order_of_magnitude_m_1line():

    # Here we write only one line to the ndt file

    import finmag

    sim = finmag.example.barmini(name='bug-saving-average-data-june-2014-c')

    sim.save_averages()

    check_magnetisation_is_of_sensible_magnitude(
        'bug_saving_average_data_june_2014_c.ndt')

#@pytest.mark.xfail
def test_ndt_writing_order_of_magnitude_m_2_and_more_lines():

    # Here we write multiple lines to the ndt file

    import finmag

    sim = finmag.example.barmini(name='bug-saving-average-data-june-2014-d')

    sim.save_averages()

    sim.schedule("save_ndt", every=10e-12)
    sim.run_until(0.1e-9)

    check_magnetisation_is_of_sensible_magnitude(
        'bug_saving_average_data_june_2014_d.ndt')

