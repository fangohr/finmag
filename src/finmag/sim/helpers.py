import numpy

def components(vs):
    """
    For a list of vectors of the form [x0, ..., xn, y0, ..., yn, z0, ..., zn]
    this will return a list of vectors with the shape
    [[x0, ..., xn], [y0, ..., yn], [z0, ..., z1]].

    """
    return vs.view().reshape((3, -1))

def vectors(vs):
    """
    For a list of vectors of the form [x0, ..., xn, y0, ..., yn, z0, ..., zn]
    this will return a list of vectors with the shape
    [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    number_of_nodes = len(vs)/3
    return vs.view().reshape((number_of_nodes, -1), order="F")

def for_dolfin(vs):
    """
    The opposite of the function vectors.

    Takes a list with the shape [[x0, y0, z0], ..., [xn, yn, zn]]
    and returns [x0, ..., xn, y0, ..., yn, z0, ..., zn].
    """
    return rows_to_columns(vs).flatten() 

def norm(v):
    """
    Returns the euclidian norm of a vector in three dimensions.

    """
    return numpy.sqrt(numpy.dot(v, v))

def normalise(vs, length=1):
    """
    Scale the vectors to the specified length.
    Expects the vectors in a list of the form [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    return numpy.array([length*v/norm(v) for v in vs])

def fnormalise(ar, length=1):
    """
    Like normalise, except it expects the arguments as an numpy.ndarray like
    dolfin provides, so [x0, ..., xn, y0, ..., yn, z0, ..., zn].

    """
    arr = components(ar)
    arr /= numpy.sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2])
    return numpy.append(arr[0],[arr[1],arr[2]]) 

def angle(v1, v2):
    """
    Returns the angle between two three-dimensional vectors.

    """
    return numpy.arccos(numpy.dot(v1, v2) / (norm(v1)*norm(v2)))

def rows_to_columns(arr):
    """
    For an array of the shape [[x1, y1, z1], ..., [xn, yn, zn]]
    returns an array of the shape [[x1, ..., xn],[y1, ..., yn],[z1, ..., zn]].

    """
    return arr.reshape(arr.size, order="F").reshape((3, -1))

def perturbed_vectors(n, direction=[1,0,0], length=1):
    """
    Returns n vectors pointing approximatively in the given direction,
    but with a random displacement. The vectors are normalised to the given
    length. The returned array looks like [[x0, y0, z0], ..., [xn, yn, zn]].

    """
    displacements = numpy.random.rand(n, 3) - 0.5
    vectors = direction + displacements
    return normalise(vectors, length)


def read_float_data(filename):
    """
    Reads floats stored in the file filename.

    Needs at least one number which can be cast to float on each line.
    Floats are to be separated by whitespace.
    Returns a list where each entry corresponds to the numbers of a line, which
    means that each entry can be itself a list.

    """
    rows = []
    with open(filename, "r") as f:
        for line in f:
            columns = [float(column) for column in line.strip().split()]
            rows.append(columns)
    return rows
