from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node:
    x_pos: float
    temperature: float
    new_temperature: float
    index: int = field(init=False)

    # Class variable to track index count
    _counter: int = 0

    def __post_init__(self):
        self.index = Node._counter
        Node._counter += 1

@dataclass
class FiniteElement:
    thermal_conductivity : float = 1
    density : float = 1
    specific_heat : float = 1
    cross_sectional_area : float = 1
    nodes : list[Node] = field(default_factory=list)
    index: int = field(init=False)

    # Class variable to track index count
    _counter: int = 0

    def __post_init__(self):
        self.index = FiniteElement._counter
        FiniteElement._counter += 1

    def length(self):
        return abs(self.nodes[1].x_pos - self.nodes[0].x_pos)

    def mass_matrix(self):
        L = self.length()
        m = (self.density * self.specific_heat * self.cross_sectional_area * L) / 6
        return np.array([
            [2 * m, 1 * m],
            [1 * m, 2 * m]
        ])

    def stiffness_matrix(self):
        L = self.length()
        k = (self.thermal_conductivity * self.cross_sectional_area) / L
        return np.array([
            [1 * k, -1 * k],
            [-1 * k, 1 * k]
        ])
