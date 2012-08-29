import dolfin as df
from finmag.util.meshes import from_geofile

def point_contacts(origins, radius, J, debug=False):
    """
    Returns a dolfin expression that locates one or more point contacts.

    The expression takes the value *J* for coordinates belonging to a point
    contact and the value 0 everywhere else. 'Belonging to a point contact'
    means that the coordinates are within *radius* of one of the
    specified *origins*.

    """
    if radius > 1e-6:
        # Assuming this is a macroscopic mesh coming from netgen, with the
        # circular part cut out at the right position.
        # Need to smooth the boundary.
        radius += 1e-4
    distance_to_origin = "sqrt(pow(x[0] - {0}, 2) + pow(x[1] - {1}, 2))"
    point_contact_conditions = ["(" + distance_to_origin.format(pos[0], pos[1]) + " <= r)" for pos in origins]
    expr_str = " || ".join(point_contact_conditions) + " ? J : 0"
    if debug:
        print expr_str
    return df.Expression(expr_str, r=radius, J=J)

if __name__ == "__main__":
    mesh = from_geofile("film.geo")
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    pc_expr = point_contacts([(-62.5, 0), (62.5, 0)], radius=10, J=1e10, debug=True)
    f = df.project(pc_expr, S1)

    df.plot(mesh)
    df.plot(f)
    df.interactive()
