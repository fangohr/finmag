#pragma once

namespace finmag {
    namespace vector {
        class vector3;
        double dot(const vector3 &r1, const vector3 &r2);

        class vector3 {
        public:
            vector3(const double R1[3], const double R2[3]) { v[0] = R1[0] - R2[0]; v[1] = R1[1] - R2[1]; v[2] = R1[2] - R2[2]; }
            vector3(const double R[3]) { v[0] = R[0]; v[1] = R[1]; v[2] = R[2]; }
            vector3(const vector3 &r1, const vector3 &r2) { v[0] = r1.v[0] - r2.v[0]; v[1] = r1.v[1] - r2.v[1]; v[2] = r1.v[2] - r2.v[2]; }
            vector3(double x, double y, double z) { v[0] = x; v[1] = y; v[2] = z; }
            vector3(const dolfin::Vertex &dv) { v[0] = dv.x(0); v[1] = dv.x(1); v[2] = dv.x(2); }
            vector3(const dolfin::Point &p) { v[0] = p.x(); v[1] = p.y(); v[2] = p.z(); }

            double length() const { return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }

            vector3 operator-(const vector3 &b) const { return vector3(v[0] - b.v[0], v[1] - b.v[1], v[2] - b.v[2]); }

            vector3 normalized() {
                double a = 1./length();
                return vector3(a*v[0], a*v[1], a*v[2]);
            }

            double x() const { return v[0]; }
            double y() const { return v[1]; }
            double z() const { return v[2]; }

            double v[3];
        };

        inline double dot(const vector3 &r1, const vector3 &r2) {
            return r1.v[0] * r2.v[0] + r1.v[1] * r2.v[1] + r1.v[2] * r2.v[2];
        }

        inline double triangle_area(const vector3 &r1, const vector3 &r2, const vector3 &r3) {
            vector3 a(r2, r1), b(r3, r1);
            double ab = dot(a, b);
            return 0.5  * sqrt(dot(a, a)*dot(b, b) - ab*ab);
        }

        inline vector3 cross(const vector3 &a, const vector3 &b) {
            return vector3(
                a.v[1]*b.v[2] - a.v[2]*b.v[1],
                a.v[2]*b.v[0] - a.v[0]*b.v[2],
                a.v[0]*b.v[1] - a.v[1]*b.v[0]
            );
        }

        inline std::ostream& operator<<(std::ostream& out, const vector3 &vec) {
            return out << "(" << vec.v[0] << " " << vec.v[1] << " " << vec.v[2] << ")";
        }

        inline double solid_angle(const vector3 &r1, const vector3 &r2, const vector3 &r3) {
            double R1_x = r1.x(), R1_y = r1.y(), R1_z = r1.z();
            double R2_x = r2.x(), R2_y = r2.y(), R2_z = r2.z();
            double R3_x = r3.x(), R3_y = r3.y(), R3_z = r3.z();
            // Wikipedia cites
            // Van Oosterom, A; Strackee, J (1983). "The Solid Angle of a Plane Triangle". IEEE Trans. Biom. Eng. BME-30 (2): 125–126. doi:10.1109/TBME.1983.325207
            // Omega = 2*atan(p/q) where
            // p = R1.R2xR3
            double p = R1_x*(R2_y*R3_z - R2_z*R3_y) - R2_x*(R1_y*R3_z - R1_z*R3_y) + R3_x*(R1_y*R2_z - R1_z*R2_y);
            // q = |R1||R2||R3| + |R3|R1.R2 + |R2|R1.R3 + |R1|R2.R3
            double R1_norm = sqrt(R1_x*R1_x + R1_y*R1_y + R1_z*R1_z);
            double R2_norm = sqrt(R2_x*R2_x + R2_y*R2_y + R2_z*R2_z);
            double R3_norm = sqrt(R3_x*R3_x + R3_y*R3_y + R3_z*R3_z);
            double R1_R2 = R1_x*R2_x + R1_y*R2_y + R1_z*R2_z;
            double R1_R3 = R1_x*R3_x + R1_y*R3_y + R1_z*R3_z;
            double R2_R3 = R2_x*R3_x + R2_y*R3_y + R2_z*R3_z;
            double q = R1_norm*R2_norm*R3_norm + R3_norm*R1_R2 + R2_norm*R1_R3 + R1_norm*R2_R3;

            double at = atan2(p, q);
            return 2*at;
        }

        inline double solid_angle(const vector3 &r, const vector3 &r1, const vector3 &r2, const vector3 &r3) {
            vector3 R1(r1, r);
            vector3 R2(r2, r);
            vector3 R3(r3, r);

            return solid_angle(R1, R2, R3);
        }
    }
}