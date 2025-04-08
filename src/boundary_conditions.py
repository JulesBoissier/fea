import numpy as np


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, node_index: int, temperature: float
):
    A[node_index, :] = 0
    A[node_index, node_index] = 1
    b[node_index] = temperature


def apply_neumann_bc(b: np.ndarray, node_index: int, flux: float, area: float = 1.0):
    b[node_index] += flux * area
