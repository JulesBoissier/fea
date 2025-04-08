import numpy as np
from src.mesh_objects import Node, FiniteElement
from src.assembly import (
    construct_global_matrices,
    construct_a_matrix,
    construct_b_vector,
)


def run_simulation(
    nodes: list[Node],
    elements: list[FiniteElement],
    delta_t: float,
    steps: int,
    bc_callback: callable = None,
):
    M, K = construct_global_matrices(nodes, elements)

    for step in range(steps):
        A = construct_a_matrix(M, K, delta_t)
        b = construct_b_vector(M, nodes, delta_t)

        if bc_callback:
            bc_callback(A, b, nodes, step)

        T_next = np.linalg.solve(A, b)

        for i, node in enumerate(nodes):
            node.temperature = T_next[i]
