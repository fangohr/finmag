"""Smoke-test the native Fredkin-Koehler BEM array entry point.

This deliberately checks the compiled ``finmag.native.llg`` implementation
used by FK demag in the pixi/FEniCS-2019 path, not the Python Magpar reference
fallback. The array interface is the compatibility surface we can keep using
when old Boost.Python DOLFIN mesh conversions are unavailable. [Codex gpt-5.5 high]
"""

import numpy as np
import dolfin as df

from finmag.native.llg import compute_bem_fk_from_arrays


def boundary_arrays(boundary_mesh):
    """Return boundary mesh arrays accepted by the native FK BEM helper."""
    entity_map = boundary_mesh.entity_map(0)
    b2g_map = entity_map.array() if hasattr(entity_map, "array") else entity_map.values()

    return (
        np.asarray(boundary_mesh.coordinates(), dtype=np.float64),
        np.asarray(boundary_mesh.cells(), dtype=np.int64),
        np.asarray(b2g_map, dtype=np.int64),
    )


def main():
    """Build a tiny boundary mesh and verify native FK BEM returns finite data."""
    mesh = df.UnitCubeMesh(1, 1, 1)
    boundary_mesh = df.BoundaryMesh(mesh, "exterior", False)
    coords, cells, b2g_map = boundary_arrays(boundary_mesh)

    bem, b2g = compute_bem_fk_from_arrays(coords, cells, b2g_map)

    if bem.shape != (len(b2g), len(b2g)):
        raise RuntimeError(
            "unexpected native FK BEM shape {} for {} boundary nodes".format(
                bem.shape, len(b2g)
            )
        )

    if not np.all(np.isfinite(bem)):
        raise RuntimeError("native FK BEM matrix contains non-finite values")

    print("native_fk_bem_shape", bem.shape)
    print("native_fk_b2g_size", len(b2g))


if __name__ == "__main__":
    main()
