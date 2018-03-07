from dolfin import *


class MyLLG(object):
    def __init__(self, V, alpha, gamma, Ms, M):
        self.V = V
        self.alpha = alpha
        self.gamma = gamma
        self.Ms = Ms
        self.M = M
        self.p = Constant(self.gamma / (1 + self.alpha ** 2))

    def H_eff(self):
        """Very temporary function to make things simple."""
        H_app = project((Constant((0, 1e5, 0))), self.V)
        # Add more effective field terms here.

        return H_app

    def variational_forms(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        self.a = inner(u, v) * dx
        self.L = inner(-self.p * cross(self.M, self.H_eff())
                   - self.p * self.alpha / self.Ms * cross(self.M, cross(self.M, self.H_eff())), v) * dx

    def compute_jacobian(self):
        self.variational_forms()
        return derivative(self.L, self.M)

L = 50e-9
mesh = BoxMesh(Point(0, 0, 0), Point(L, L, L), 5, 5, 5)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
M = project((Constant((8e6, 0, 0))), V)
llg = MyLLG(V=V, alpha=0.5, gamma=2.211e5, Ms=8e6, M=M)

jacobian = llg.compute_jacobian()
