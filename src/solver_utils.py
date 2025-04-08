import numpy as np

from src.mesh_objects import Node, FiniteElement


def construct_global_matrices(nodes: list[Node], elements: list[FiniteElement]):

    size = len(nodes)
    global_mass_matrix = np.zeros((size, size))
    global_stiffness_matrix = np.zeros((size, size))

    for element in elements:
        local_mass = element.mass_matrix()
        local_stiffness = element.stiffness_matrix()

        # Use node.index directly as global indices
        indices = [node.index for node in element.nodes]

        for i_local, i_global in enumerate(indices):
            for j_local, j_global in enumerate(indices):
                global_mass_matrix[i_global, j_global] += local_mass[i_local, j_local]
                global_stiffness_matrix[i_global, j_global] += local_stiffness[
                    i_local, j_local
                ]

    return global_mass_matrix, global_stiffness_matrix


def construct_a_matrix(
    global_mass_matrix: np.ndarray, global_stiffness_matrix: np.ndarray, delta_t: float
) -> np.ndarray:
    """
    Constructs the A matrix for Backward Euler time stepping:
    A = M / delta_t + K
    """
    return global_mass_matrix / delta_t + global_stiffness_matrix


def construct_b_vector(
    global_mass_matrix: np.ndarray, nodes: list[Node], delta_t: float
) -> np.ndarray:
    """
    Constructs the b vector: b = M / delta_t * T^n
    """
    T_n = np.array([node.temperature for node in nodes])
    return global_mass_matrix @ (T_n / delta_t)


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, node_index: int, temperature: float
):
    """
    Applies a Dirichlet BC (fixed temperature) at a node by modifying A and b in-place.

    Args:
        A: Global system matrix (modified in-place)
        b: Global RHS vector (modified in-place)
        node_index: Index of the node where temperature is fixed
        temperature: The fixed temperature value (°C)
    """
    A[node_index, :] = 0
    A[node_index, node_index] = 1
    b[node_index] = temperature


def apply_neumann_bc(b: np.ndarray, node_index: int, flux: float, area: float = 1.0):
    """
    Applies a Neumann BC (heat flux) at a node by modifying the RHS vector.

    Args:
        b: Global RHS vector (modified in-place)
        node_index: Index of the node where flux is applied
        flux: Heat flux in W/m² (positive = heat in, negative = heat out)
        area: Surface area over which flux is applied (default = 1 m² for 1D)
    """
    b[node_index] += flux * area
