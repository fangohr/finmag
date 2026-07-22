# FinMag - DOLFINx port
# CONTACT: h.fangohr@soton.ac.uk
"""Netgen ``algebraic3d`` CSG loader for the DOLFINx port (Task 30 amendment).

Task 18 deferred ``finmag.util.meshes.from_geofile`` / ``from_csg`` because the
legacy path went ``.geo`` -> Netgen -> DIFFPACK -> dolfin-XML, none of which is
available in the DOLFINx env.  The examples slice (Task 30) needs real legacy
``.geo`` files to load with *zero* changes to the example scripts, so this module
lifts that deferral for the Netgen-CSG **subset the examples actually use**,
building the geometry through the same Task 18 OCC/Gmsh machinery
(``finmag.util.meshes._mesh_from_geometry``) that the named generators use.

Supported dialect (see the Task 30 report for the per-file inventory)::

    solid <name> = <expr> [ -maxh = <h> ] ;
    tlo   <name>          [ -maxh = <h> ] ;

    <expr>      := <term> ( ('and' 'not' | 'and' | 'or') <term> )*
    <term>      := <primitive> | <name-ref> | '(' <expr> ')' | 'not' <term>
    <primitive> := orthobrick '(' p ';' p ')'            -> OCC box
                 | sphere     '(' p ';' r ')'            -> OCC sphere
                 | cylinder   '(' p ';' p ';' r ')'      -> finite OCC cylinder
                 | plane      '(' p ';' n ')'            -> axis-aligned half-space

Booleans map to OCC operations: ``and`` -> intersect/common, ``and not`` -> cut,
``or`` -> fuse.  A group of axis-aligned ``plane`` half-spaces that closes an
axis-aligned box is recognised as that box (this is how Netgen ``.geo`` files
commonly spell a cuboid, e.g. ``bar30_30_100.geo``).  Multiple ``tlo`` objects
are fused into a single-material mesh (subdomain markers are NOT preserved --
that is the multi-region case explicitly outside this single-material slice).

Netgen's ``cylinder`` is formally infinite and bounded by capping planes; here
the two axis points are taken as the finite extent and axis-aligned capping
planes that coincide with those ends are treated as redundant (this matches the
common idiom, e.g. ``film.geo``).  Any construct outside this subset --
``multitranslate``, non-axis-aligned planes, ellipsoids, ... -- raises
``NotImplementedError`` naming the construct (fail-forward for future files).

The netgen *binary* backend stays deferred; this is a from-text CSG loader only.
[Claude Opus 4.8]
"""

import os
import re

__all__ = ["from_geofile", "from_csg"]


class GeoFileError(ValueError):
    """Raised for a malformed or unsupported Netgen-CSG construct."""


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

# Strip Netgen '#' comments, then tokenise into words / numbers / punctuation.
_TOKEN_RE = re.compile(r"""
      (?P<maxh>-maxh)
    | (?P<number>[-+]?\d+(\.\d*)?([eE][-+]?\d+)?)
    | (?P<name>[A-Za-z_]\w*)
    | (?P<punct>[();,=])
""", re.VERBOSE)


def _tokenize(text):
    # remove comments (# to end of line)
    text = re.sub(r"#[^\n]*", " ", text)
    tokens = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            if text[pos].isspace():
                pos += 1
                continue
            raise GeoFileError(
                "unexpected character {!r} in .geo/CSG text".format(text[pos]))
        pos = m.end()
        if m.lastgroup == "number":
            tokens.append(("number", float(m.group())))
        elif m.lastgroup == "maxh":
            tokens.append(("maxh", "-maxh"))
        elif m.lastgroup == "name":
            tokens.append(("name", m.group()))
        else:
            tokens.append((m.group(), m.group()))
    return tokens


# --------------------------------------------------------------------------
# AST nodes -- each .emit(occ) returns a list of OCC volume tags
# --------------------------------------------------------------------------

class _Node(object):
    is_plane = False


class _Orthobrick(_Node):
    def __init__(self, p0, p1):
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)

    def emit(self, occ):
        (x0, y0, z0), (x1, y1, z1) = self.p0, self.p1
        return [occ.addBox(min(x0, x1), min(y0, y1), min(z0, z1),
                           abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))]


class _Sphere(_Node):
    def __init__(self, center, r):
        self.center = tuple(center)
        self.r = float(r)

    def emit(self, occ):
        cx, cy, cz = self.center
        return [occ.addSphere(cx, cy, cz, self.r)]


class _Cylinder(_Node):
    # Netgen cylinder is infinite; the two axis points are taken as the finite
    # extent (redundant axis-aligned capping planes in the same 'and' chain are
    # dropped -- see module docstring).
    def __init__(self, p0, p1, r):
        self.p0 = tuple(p0)
        self.p1 = tuple(p1)
        self.r = float(r)

    def emit(self, occ):
        x0, y0, z0 = self.p0
        x1, y1, z1 = self.p1
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        if dx == dy == dz == 0:
            raise GeoFileError("degenerate cylinder (coincident axis points)")
        return [occ.addCylinder(x0, y0, z0, dx, dy, dz, self.r)]


class _Plane(_Node):
    is_plane = True

    def __init__(self, point, normal):
        self.point = tuple(point)
        self.normal = tuple(normal)

    def axis_bound(self):
        """Return ``(axis, sign, value)`` for an axis-aligned plane.

        Netgen ``plane(p; n)`` keeps the half-space on the ``-n`` side of ``p``.
        A negative-component normal therefore gives a *lower* bound, a positive
        one an *upper* bound, along ``axis``.
        """
        nz = [i for i, c in enumerate(self.normal) if c != 0.0]
        if len(nz) != 1:
            raise NotImplementedError(
                "from_geofile: only axis-aligned 'plane' half-spaces are "
                "supported (got normal {}). Oblique planes are not ported."
                .format(self.normal))
        axis = nz[0]
        sign = 1 if self.normal[axis] > 0 else -1
        return axis, sign, self.point[axis]


class _Not(_Node):
    def __init__(self, operand):
        self.operand = operand


class _And(_Node):
    def __init__(self, operands):
        self.operands = operands

    def emit(self, occ):
        planes = [o for o in self.operands if getattr(o, "is_plane", False)]
        nots = [o for o in self.operands if isinstance(o, _Not)]
        solids = [o for o in self.operands
                  if not getattr(o, "is_plane", False) and not isinstance(o, _Not)]

        # A cylinder present in the chain is already finite (its two axis points
        # bound it). Netgen still spells the caps as explicit half-spaces, so a
        # capping plane that coincides with a cylinder axis endpoint is redundant
        # and dropped. A plane offset from those endpoints would actually
        # truncate the cylinder -- that construct is not ported, so fail forward
        # (name it) rather than silently ignoring the truncation.
        cylinders = [s for s in solids if isinstance(s, _Cylinder)]
        if cylinders and planes:
            _validate_cylinder_caps(cylinders, planes)
            planes = []

        if planes:
            solids = solids + [_box_from_planes(planes)]

        if not solids:
            raise GeoFileError("'and' expression has no bounded solid")

        tags = solids[0].emit(occ)
        for s in solids[1:]:
            other = s.emit(occ)
            tags, _ = occ.intersect([(3, t) for t in tags],
                                    [(3, t) for t in other])
            tags = [t for (_d, t) in tags]

        for nt in nots:
            other = nt.operand.emit(occ)
            tags, _ = occ.cut([(3, t) for t in tags],
                              [(3, t) for t in other])
            tags = [t for (_d, t) in tags]
        return tags


class _Or(_Node):
    def __init__(self, operands):
        self.operands = operands

    def emit(self, occ):
        tags = self.operands[0].emit(occ)
        for o in self.operands[1:]:
            other = o.emit(occ)
            tags, _ = occ.fuse([(3, t) for t in tags],
                               [(3, t) for t in other])
            tags = [t for (_d, t) in tags]
        return tags


def _cylinder_axis_extent(cyl):
    """Return ``(axis, lo, hi, length)`` for an axis-aligned cylinder."""
    d = tuple(b - a for a, b in zip(cyl.p0, cyl.p1))
    nz = [i for i, c in enumerate(d) if c != 0.0]
    if len(nz) != 1:
        raise NotImplementedError(
            "from_geofile: only axis-aligned cylinders can have their capping "
            "planes validated (got axis direction {}). Oblique cylinders are "
            "not ported.".format(d))
    axis = nz[0]
    lo = min(cyl.p0[axis], cyl.p1[axis])
    hi = max(cyl.p0[axis], cyl.p1[axis])
    return axis, lo, hi, hi - lo


def _validate_cylinder_caps(cylinders, planes):
    """Confirm each dropped capping plane coincides with a cylinder end.

    A plane is a redundant cap only if its axis is parallel to a cylinder's axis
    and its bound coincides (within a length-relative tolerance) with one of that
    cylinder's two axis endpoints. Any other plane truncates the cylinder away
    from its ends -- a construct outside the ported subset -- so it raises
    ``NotImplementedError`` naming ``offset-plane-truncated-cylinder``.
    """
    extents = [_cylinder_axis_extent(cyl) for cyl in cylinders]
    for plane in planes:
        axis, _sign, value = plane.axis_bound()
        coincides = False
        for caxis, lo, hi, length in extents:
            if caxis != axis:
                continue
            tol = 1e-9 * max(abs(length), 1.0)
            if abs(value - lo) <= tol or abs(value - hi) <= tol:
                coincides = True
                break
        if not coincides:
            raise NotImplementedError(
                "from_geofile: an offset-plane-truncated-cylinder construct was "
                "found (a 'plane' half-space bounded at {:g} along axis {} does "
                "not coincide with any cylinder axis endpoint). Capping planes "
                "that truncate a cylinder away from its axis endpoints are not "
                "ported.".format(value, axis))


def _box_from_planes(planes):
    lo = {}
    hi = {}
    for p in planes:
        axis, sign, value = p.axis_bound()
        if sign < 0:
            lo[axis] = value
        else:
            hi[axis] = value
    if set(lo) != {0, 1, 2} or set(hi) != {0, 1, 2}:
        raise NotImplementedError(
            "from_geofile: a group of 'plane' half-spaces was found that does "
            "not close an axis-aligned box (lower bounds {}, upper bounds {}). "
            "Only box-forming plane groups are supported.".format(
                sorted(lo), sorted(hi)))
    p0 = (lo[0], lo[1], lo[2])
    p1 = (hi[0], hi[1], hi[2])
    return _Orthobrick(p0, p1)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

class _Parser(object):
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0
        self.solids = {}          # name -> (node, maxh_or_None)
        self.tlos = []            # list of (name, maxh_or_None)

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def _next(self):
        tok = self._peek()
        self.i += 1
        return tok

    def _expect(self, kind):
        tok = self._next()
        if tok[0] != kind:
            raise GeoFileError("expected {!r} but got {!r}".format(kind, tok[1]))
        return tok

    def _number(self):
        tok = self._next()
        if tok[0] != "number":
            raise GeoFileError("expected a number but got {!r}".format(tok[1]))
        return tok[1]

    def _point(self):
        x = self._number()
        self._expect(",")
        y = self._number()
        self._expect(",")
        z = self._number()
        return (x, y, z)

    def parse(self):
        # optional leading 'algebraic3d' (Netgen keywords are case-insensitive)
        peeked = self._peek()
        if peeked[0] == "name" and peeked[1].lower() == "algebraic3d":
            self._next()
        while self.i < len(self.toks):
            kind, val = self._peek()
            if kind == "name" and val.lower() == "solid":
                self._parse_solid()
            elif kind == "name" and val.lower() == "tlo":
                self._parse_tlo()
            else:
                raise GeoFileError(
                    "expected 'solid' or 'tlo' but got {!r}".format(val))
        return self

    def _parse_solid(self):
        self._next()  # 'solid'
        # Netgen identifiers are case-insensitive; store/look up folded.
        name = self._expect("name")[1].lower()
        self._expect("=")
        node = self._parse_expr()
        maxh = self._maybe_maxh()
        self._expect(";")
        self.solids[name] = (node, maxh)

    def _parse_tlo(self):
        self._next()  # 'tlo'
        name = self._expect("name")[1].lower()
        maxh = self._maybe_maxh()
        self._expect(";")
        self.tlos.append((name, maxh))

    def _maybe_maxh(self):
        if self._peek()[0] == "maxh":
            self._next()
            self._expect("=")
            return self._number()
        return None

    # expr := term ( ('and' ['not'] | 'or') term )*
    def _parse_expr(self):
        left = self._parse_term()
        and_ops = [left]
        or_ops = []
        mode = "and"
        while True:
            kind, val = self._peek()
            if kind == "name" and val.lower() == "and":
                self._next()
                nxt = self._peek()
                if nxt[0] == "name" and nxt[1].lower() == "not":
                    self._next()
                    and_ops.append(_Not(self._parse_term()))
                else:
                    and_ops.append(self._parse_term())
            elif kind == "name" and val.lower() == "or":
                self._next()
                # flush current 'and' group as one operand of the 'or'
                or_ops.append(self._collapse_and(and_ops))
                and_ops = [self._parse_term()]
                mode = "or"
            else:
                break
        group = self._collapse_and(and_ops)
        if or_ops:
            or_ops.append(group)
            return _Or(or_ops)
        return group

    @staticmethod
    def _collapse_and(and_ops):
        # A single operand is returned as-is (including a lone _Not, so a parent
        # 'and' sees it as a cut; a truly top-level lone 'not' is unbounded and
        # is rejected later by _And.emit's "no bounded solid" check).
        if len(and_ops) == 1:
            return and_ops[0]
        return _And(and_ops)

    def _parse_term(self):
        kind, val = self._peek()
        if kind == "(":
            self._next()
            node = self._parse_expr()
            self._expect(")")
            return node
        low = val.lower() if kind == "name" else val
        if kind == "name" and low == "not":
            self._next()
            return _Not(self._parse_term())
        if kind == "name" and low in _PRIMITIVES:
            self._next()  # consume the primitive keyword
            return _PRIMITIVES[low](self)
        if kind == "name" and low in ("multitranslate", "multitranslatexyz",
                                      "ellipsoid", "cone", "revolution",
                                      "extrusion", "torus", "polyhedron"):
            raise NotImplementedError(
                "from_geofile: the Netgen CSG construct '{}' is not supported "
                "by the DOLFINx port's CSG subset.".format(val))
        if kind == "name":
            # reference to a previously named solid (case-insensitive)
            self._next()
            if low not in self.solids:
                raise GeoFileError("unknown solid reference {!r}".format(val))
            return self.solids[low][0]
        raise GeoFileError("unexpected token {!r} in expression".format(val))


def _parse_orthobrick(p):
    p._expect("(")
    a = p._point()
    p._expect(";")
    b = p._point()
    p._expect(")")
    return _Orthobrick(a, b)


def _parse_sphere(p):
    p._expect("(")
    c = p._point()
    p._expect(";")
    r = p._number()
    p._expect(")")
    return _Sphere(c, r)


def _parse_cylinder(p):
    p._expect("(")
    a = p._point()
    p._expect(";")
    b = p._point()
    p._expect(";")
    r = p._number()
    p._expect(")")
    return _Cylinder(a, b, r)


def _parse_plane(p):
    p._expect("(")
    point = p._point()
    p._expect(";")
    normal = p._point()
    p._expect(")")
    return _Plane(point, normal)


_PRIMITIVES = {
    "orthobrick": _parse_orthobrick,
    "sphere": _parse_sphere,
    "cylinder": _parse_cylinder,
    "plane": _parse_plane,
}


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def _build_mesh(parser, csg_key, maxh, save_result, filename, directory):
    from finmag.util.meshes import _mesh_from_geometry

    if not parser.tlos:
        raise GeoFileError("no 'tlo' (top-level object) found")

    # effective maxh per tlo: tlo-level overrides the solid-level value
    effective = []
    for name, tlo_maxh in parser.tlos:
        if name not in parser.solids:
            raise GeoFileError("tlo references unknown solid {!r}".format(name))
        node, solid_maxh = parser.solids[name]
        h = tlo_maxh if tlo_maxh is not None else solid_maxh
        effective.append(h)

    hs = [h for h in effective if h is not None]
    if maxh is None:
        maxh = min(hs) if hs else None
    if maxh is None:
        raise GeoFileError(
            "no maxh specified in the .geo/CSG text or the call")

    tlo_nodes = [parser.solids[name][0] for name, _ in parser.tlos]

    def add_solids(occ):
        all_tags = []
        for node in tlo_nodes:
            all_tags.extend(node.emit(occ))
        if len(all_tags) > 1:
            # fuse all top-level objects into one single-material domain
            fused, _ = occ.fuse([(3, all_tags[0])],
                                [(3, t) for t in all_tags[1:]])
            all_tags = [t for (_d, t) in fused]
        return all_tags

    return _mesh_from_geometry(csg_key, add_solids, maxh, save_result,
                               filename, directory)


def from_csg(csg_string, save_result=True, filename='', directory='', *, maxh=None):
    """Build a ``dolfinx.mesh`` from a Netgen ``algebraic3d`` CSG string.

    Supports the subset documented in this module's docstring. ``maxh`` (if
    given) overrides any ``-maxh`` in the text. Caching mirrors the legacy
    contract: the CSG text is the md5 cache key when ``filename`` is empty.

    ``save_result`` keeps the legacy positional slot (legacy ``from_csg`` was
    ``from_csg(csg, save_result=True, filename='', directory='')``); ``maxh`` is
    the port-only extension and is therefore keyword-only, so a legacy positional
    ``from_csg(text, False)`` binds ``save_result`` as before.
    """
    parser = _Parser(_tokenize(csg_string)).parse()
    return _build_mesh(parser, csg_string, maxh, save_result, filename, directory)


def from_geofile(geofile, save_result=True, filename='', directory='', *, maxh=None):
    """Build a ``dolfinx.mesh`` from a Netgen ``.geo`` file.

    Ported subset of the legacy ``from_geofile`` (Task 18 deferral partially
    lifted, Task 30). The mesh is cached next to the ``.geo`` file (keyed on the
    file's *content*, mirroring the legacy behaviour of caching a compiled mesh
    beside the geometry); pass ``save_result=False`` to skip the cache.

    ``save_result`` keeps the legacy positional slot (legacy ``from_geofile`` was
    ``from_geofile(geofile, save_result=True)``); ``maxh`` is the port-only
    extension and is therefore keyword-only, so a legacy positional
    ``from_geofile(f, False)`` binds ``save_result`` as before.
    """
    with open(geofile, "r") as f:
        text = f.read()
    if directory == '' and save_result:
        directory = os.path.dirname(os.path.abspath(geofile)) or os.curdir
    parser = _Parser(_tokenize(text)).parse()
    # Key the cache on the file content (legacy keyed the compiled mesh on the
    # geometry text), so edits to the .geo invalidate the cache automatically.
    return _build_mesh(parser, text, maxh, save_result, filename, directory)
