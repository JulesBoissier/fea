from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Node:
    x_pos: float
    y_pos: float
    temperature: float
    new_temperature: float = None
    index: int = field(init=False)

    _counter: int = 0

    def __post_init__(self):
        self.index = Node._counter
        Node._counter += 1


@dataclass
class FiniteElement(ABC):
    thermal_conductivity: float = 1
    density: float = 1
    specific_heat: float = 1
    # cross_sectional_area: float = 1
    nodes: list[Node] = field(default_factory=list)
    index: int = field(init=False)

    _counter: int = 0

    def __post_init__(self):
        self.index = FiniteElement._counter
        FiniteElement._counter += 1

    def material_matrix(self):
        """Return 2D isotropic thermal conductivity matrix."""
        k = self.thermal_conductivity
        return k * np.eye(2)

    @abstractmethod
    def shape_functions(self, x_local: float, y_local: float):
        pass

    @abstractmethod
    def dN_dlocal(self, x_local: float, y_local: float):
        pass

    @abstractmethod
    def jacobian(self, x_local: float, y_local: float):
        pass

    @abstractmethod
    def stiffness_matrix(self):
        pass

    @abstractmethod
    def mass_matrix(self):
        pass


@dataclass
class TriElement(FiniteElement):
    def shape_functions(self, x_local, y_local):
        return np.array([1 - x_local - y_local, x_local, y_local])

    def dN_dlocal(self, x_local, y_local):
        return np.array(
            [
                [-1, -1],
                [1, 0],
                [0, 1],
            ]
        )

    def jacobian(self, x_local=0, y_local=0):
        """Constant Jacobian for linear triangle."""
        dN = self.dN_dlocal(x_local, y_local)
        coords = np.array([[n.x_pos, n.y_pos] for n in self.nodes])
        return dN.T @ coords

    def area(self):
        x = [n.x_pos for n in self.nodes]
        y = [n.y_pos for n in self.nodes]
        return 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def stiffness_matrix(self):
        J = self.jacobian()
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Element {self.index} has invalid geometry.")

        invJ = np.linalg.inv(J)
        dN = self.dN_dlocal(0, 0)
        B = dN @ invJ
        D = self.material_matrix()
        return 0.5 * detJ * (B @ D @ B.T)

    def mass_matrix(self):
        A = self.area()
        p = self.density
        c = self.specific_heat
        m = (p * c * A) / 12
        return m * (np.ones((3, 3)) + np.eye(3))


@dataclass
class QuadElement(FiniteElement):
    def shape_functions(self, x_local, y_local):
        return np.array(
            [
                0.25 * (1 - x_local) * (1 - y_local),
                0.25 * (1 + x_local) * (1 - y_local),
                0.25 * (1 + x_local) * (1 + y_local),
                0.25 * (1 - x_local) * (1 + y_local),
            ]
        )

    def dN_dlocal(self, x_local, y_local):
        return (
            np.array(
                [
                    [-(1 - y_local), -(1 - x_local)],
                    [(1 - y_local), -(1 + x_local)],
                    [(1 + y_local), (1 + x_local)],
                    [-(1 + y_local), (1 - x_local)],
                ]
            )
            * 0.25
        )

    def jacobian(self, x_local, y_local):
        dN = self.dN_dlocal(x_local, y_local)
        coords = np.array([[n.x_pos, n.y_pos] for n in self.nodes])
        return dN.T @ coords

    def stiffness_matrix(self):
        gauss_pts = [
            (-1 / np.sqrt(3), -1 / np.sqrt(3)),
            (1 / np.sqrt(3), -1 / np.sqrt(3)),
            (1 / np.sqrt(3), 1 / np.sqrt(3)),
            (-1 / np.sqrt(3), 1 / np.sqrt(3)),
        ]

        K = np.zeros((4, 4))
        D = self.material_matrix()

        for x_local, y_local in gauss_pts:
            J = self.jacobian(x_local, y_local)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(
                    f"Element {self.index} has invalid geometry at Gauss point."
                )

            invJ = np.linalg.inv(J)
            dN = self.dN_dlocal(x_local, y_local)
            B = dN @ invJ
            K += (B @ D @ B.T) * detJ

        return K

    def mass_matrix(self):
        gauss_pts = [
            (-1 / np.sqrt(3), -1 / np.sqrt(3)),
            (1 / np.sqrt(3), -1 / np.sqrt(3)),
            (1 / np.sqrt(3), 1 / np.sqrt(3)),
            (-1 / np.sqrt(3), 1 / np.sqrt(3)),
        ]

        M = np.zeros((4, 4))
        p = self.density
        c = self.specific_heat

        for x_local, y_local in gauss_pts:
            N = self.shape_functions(x_local, y_local).reshape((4, 1))
            J = self.jacobian(x_local, y_local)
            detJ = np.linalg.det(J)
            M += (N @ N.T) * detJ

        return p * c * M
