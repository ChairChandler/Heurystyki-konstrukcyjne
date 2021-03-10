import numpy as np
from typing import List


def nng(distance_matrix: np.ndarray, vertices_percent=1, start_vertex=0) -> (float, List[int]):
    vertices_amount = distance_matrix.shape[0]
    min_vertices_amount = int(np.ceil(vertices_percent * vertices_amount))

    cost, distance_matrix[:, start_vertex] = 0, np.inf
    route, unvisited = [start_vertex], {i for i in range(vertices_amount) if i != start_vertex}
    init_vertex_distances = distance_matrix[start_vertex].copy()

    for _ in range(1, min_vertices_amount):
        distances = distance_matrix[start_vertex]
        nearest = np.argmin(distances)
        cost += distances[nearest]
        route.append(nearest)
        unvisited.remove(nearest)
        start_vertex = nearest
        distance_matrix[:, nearest] = np.inf

    cost += init_vertex_distances[route[-1]]
    route.append(route[0])

    return int(cost), route
