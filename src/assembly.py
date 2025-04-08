import numpy as np
from src.mesh_objects import Node, FiniteElement


def construct_global_matrices(nodes: list[Node], elements: list[FiniteElement]):
    size = len(nodes)
    M = np.zeros((size, size))
    K = np.zeros((size, size))

    for element in elements:
        local_M = element.mass_matrix()
        local_K = element.stiffness_matrix()
        indices = [node.index for node in element.nodes]

        for i_local, i_global in enumerate(indices):
            for j_local, j_global in enumerate(indices):
                M[i_global, j_global] += local_M[i_local, j_local]
                K[i_global, j_global] += local_K[i_local, j_local]

    return M, K


def construct_a_matrix(M: np.ndarray, K: np.ndarray, delta_t: float) -> np.ndarray:
    return M / delta_t + K


def construct_b_vector(M: np.ndarray, nodes: list[Node], delta_t: float) -> np.ndarray:
    T_n = np.array([node.temperature for node in nodes])
    return M @ (T_n / delta_t)
