import numpy

def components(vs):
    """
    For a list of vectors of the form [x0, ..., xn, y0, ..., yn, z0, ..., zn]
    this will return a list of vectors with the shape
    [[x0, ..., xn], [y0, ..., yn], [z0, ..., z1]].

    """
    return vs.view().reshape((3, -1))

def make_vectors_function(vs):
    """
    Creates a function which extracts the vectors out of vector data like
    dolfin provides it. It will need an example of such a vector to infer
    the number of nodes.

    """
    number_of_nodes = len(vs)/3
    def vectors(vs):
        """
        For a list of vectors of shape [x0, ..., xn, y0, ..., yn, z0, ..., zn]
        this will return a list of vectors with the shape
        [[x0, y0, z0], ..., [xn, yn, zn]]
    
        """
        return vs.view().reshape((number_of_nodes, -1), order="F")
    return vectors

def norm(v):
    """
    Returns the euclidian norm of a vector in three dimensions.

    """
    return numpy.sqrt(numpy.dot(v, v))

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

