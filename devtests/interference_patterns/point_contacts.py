import dolfin as df

def point_contacts(origins, radius, J, debug=False):
    """
    Returns a dolfin expression that locates one or more point contacts.

    The expression takes the value *J* for coordinates belonging to a point
    contact and the value 0 everywhere else. 'Belonging to a point contact'
    means that the coordinates are within *radius* of one of the
    specified *origins*.

    """
    distance_to_origin = "sqrt(pow(x[0] - {0}, 2) + pow(x[1] - {1}, 2))"
    point_contact_conditions = ["(" + distance_to_origin.format(pos[0], pos[1]) + " <= r)" for pos in origins]
    expr_str = " || ".join(point_contact_conditions) + " ? J : 0"
    if debug:
        print expr_str
    return df.Expression(expr_str, r=radius, J=J)

if __name__ == "__main__":
    mesh = df.Box(0, 0, 0, 70e-9, 50e-9, 1e-9, 240, 200, 1)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    pc_expr = point_contacts([(20e-9, 25e-9), (50e-9, 25e-9)], radius=10e-9, J=1e10, debug=True)
    f = df.project(pc_expr, S1)

    df.plot(f)
    df.interactive()
