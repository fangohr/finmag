import instant
import numpy as np

c_code = """
void double_entries(int a_n, double *a) {
    for ( int i=0; i<a_n; i++ ) {
        a[i] *= 2;
    }
}
"""
double_entries = instant.inline_with_numpy(c_code, arrays=[["a_n", "a"]])

my_array = np.array([1.0, 2.0, 3.0])
print "\nBefore:\n", my_array
double_entries(my_array)
print "After calling double_entries:\n", my_array

assert np.array_equal(my_array, np.array([2.0, 4.0, 6.0]))

# ALTERNATIVE
# provide a new array to contain the result

c_code_alt = """
void double_entries_alt(int a_n, double *a, int b_n, double *b) {
    assert ( a_n == b_n );
    for ( int i=0; i<a_n; i++ ) {
        b[i] = 2 * a[i];
    }
}
"""
args = [["a_n", "a"], ["b_n", "b"]]
double_entries_alt = instant.inline_with_numpy(c_code_alt, arrays=args)

my_new_array = np.array([1.0, 2.0, 3.0])
my_doubled_array = np.zeros(len(my_new_array))
print "\nAlternative.\nBefore:\n", my_new_array, my_doubled_array
double_entries_alt(my_new_array, my_doubled_array)
print "After calling double_entries_alt:\n", my_new_array, my_doubled_array

assert np.array_equal(my_new_array, np.array([1.0, 2.0, 3.0]))
assert np.array_equal(my_doubled_array, np.array([2.0, 4.0, 6.0]))

###########################################################################
#Gabriel @Mark Thanks for the help so far here my attempt with restrict_to
###########################################################################

c_code_restrict_to = """
void restrict_to(int bigvec_n, double *bigvec, int resvec_n, double *resvec, int dofs_n, long *dofs) {
    for ( int i=0; i<resvec_n; i++ )
        { resvec[i] = bigvec[dofs[i]]; }
}
"""

args = [["bigvec_n", "bigvec"],["resvec_n", "resvec"],["dofs_n","dofs", "long"]]
restrict_to = instant.inline_with_numpy(c_code_restrict_to, arrays=args)


#Test restrict_to
bigvec = np.array([4.0,5.0,6.0,7.0,8.0])
resvec = np.zeros(2)
dofs = np.array([0,3])
print dofs.dtype.name
print "\n resvec before: \n", resvec
restrict_to(bigvec,resvec,dofs)
print "resvec after: \n", resvec
