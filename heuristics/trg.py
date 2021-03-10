import numpy as np
from typing import List
import heapq


def trg(distance_matrix: np.ndarray, vertices_percent=1, start_vertex=0, k_regret_coeff=2) -> (float, List[int]):
    vertices_amount = distance_matrix.shape[0]
    min_vertices_amount = int(np.ceil(vertices_percent * vertices_amount))

    route, unvisited = [start_vertex, start_vertex], {i for i in range(vertices_amount) if i != start_vertex}
    cost = 0

    # find nn for both vertices
    for _ in range(1, min_vertices_amount):
        candidates = []
        for c_vid in unvisited:
            k_regret = []
            for a_vid, b_vid in zip(route, route[1:]):
                a_to_b = distance_matrix[a_vid][b_vid]
                a_to_c = distance_matrix[a_vid][c_vid]
                c_to_b = distance_matrix[c_vid][b_vid]
                local_cost = a_to_c + c_to_b
                diff = local_cost - (a_to_b if a_to_b not in [-np.inf, np.inf] else 0)

                k_regret.append((cost + diff, a_vid))

            k_regret.sort()
            k_regret = k_regret[:k_regret_coeff]
            k_regret_min, vertex = min(k_regret)
            if len(k_regret) > 1:
                k_regret_sum = sum(cost - k_regret_min for cost, _ in k_regret)
                heapq.heappush(candidates, (-k_regret_sum, k_regret_min, vertex, c_vid))
            else:
                heapq.heappush(candidates, (k_regret_min, k_regret_min, vertex, c_vid))

        _, min_cost, source, dest = heapq.heappop(candidates)
        route.insert(route.index(source) + 1, dest)
        unvisited.remove(dest)
        cost = min_cost

    return int(cost), route
