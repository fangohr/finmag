import dolfin as df

def point_contacts(origins, radius, value, debug=False):
    """
    Returns a dolfin expression that locates one or more point contacts.

    The expression takes the value *value* for coordinates belonging to a point
    contact and the value 0 everywhere else. 'Belonging to a point contact'
    means that the coordinates are within *radius* of one of the
    specified *origins*.

    """
    distance_to_origin = "sqrt(pow(x[0] - {0}, 2) + pow(x[1] - {1}, 2))"
    point_contact_conditions = ["(" + distance_to_origin.format(pos[0], pos[1]) + " <= r)" for pos in origins]
    expr_str = " || ".join(point_contact_conditions) + " ? 1 : 0"
    if debug:
        print expr_str
    return df.Expression(expr_str, r=radius)

if __name__ == "__main__":
    mesh = df.Box(0, 0, 0, 70, 50, 1, 240, 200, 1)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    pc_expr = point_contacts([(20, 25), (50, 25)], radius=10, value=1, debug=True)
    f = df.project(pc_expr, S1)

    df.plot(f)
    df.interactive()
