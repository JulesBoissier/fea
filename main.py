from src.mesh_objects import Node, FiniteElement
from src.solver import run_simulation
from src.boundary_conditions import apply_neumann_bc


def insulated_boundaries_bc(A, b, nodes, step):
    apply_neumann_bc(b, 0, 0)
    apply_neumann_bc(b, 2, 0)


def main():
    node_1 = Node(x_pos=0, temperature=100)
    node_2 = Node(x_pos=1, temperature=100)
    node_3 = Node(x_pos=2, temperature=0)

    ele_1 = FiniteElement(nodes=[node_1, node_2])
    ele_2 = FiniteElement(nodes=[node_2, node_3])

    nodes = [node_1, node_2, node_3]
    elements = [ele_1, ele_2]

    run_simulation(
        nodes, elements, delta_t=0.01, steps=500, bc_callback=insulated_boundaries_bc
    )

    print("Temperatures:\n")
    for node in nodes:
        print(node.temperature)


if __name__ == "__main__":
    main()
