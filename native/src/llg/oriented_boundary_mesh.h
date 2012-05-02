#pragma once

namespace finmag { namespace llg {
    class OrientedBoundaryMesh : public dolfin::BoundaryMesh {
    public:
        OrientedBoundaryMesh(const boost::shared_ptr<dolfin::Mesh> &mesh_ptr);

        dolfin::MeshFunction<unsigned int>& cell_map() { return _cell_map; }
        const dolfin::MeshFunction<unsigned int>& cell_map() const { return _cell_map; }
        dolfin::MeshFunction<unsigned int>& vertex_map() { return _vertex_map; }
        const dolfin::MeshFunction<unsigned int>& vertex_map() const { return _vertex_map; }
    private:
        // hide copy constructor and assignment
        OrientedBoundaryMesh(const OrientedBoundaryMesh&);
        void operator=(const OrientedBoundaryMesh&);

        dolfin::MeshFunction<unsigned int> _cell_map;
        dolfin::MeshFunction<unsigned int> _vertex_map;
    };

    void register_oriented_boundary_mesh();
}}