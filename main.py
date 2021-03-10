from csv import reader
from pathlib import Path
from shutil import Error
from statistics import mean

import heuristics as heu
from typing import List
from sys import argv
from json import dump as json_dump
import matplotlib.pyplot as plt
import numpy as np


def check_dir(path: Path):
    abs_path = path.absolute()
    if not path.exists():
        raise FileNotFoundError(f'Directory {abs_path} does not exists')
    elif not path.is_dir():
        raise Error(f'Expected directory ({abs_path})')


def create_distance_matrix(points) -> np.ndarray:
    neighbors_amount = points[-1][0]
    neighbors_matrix = np.ndarray(shape=(neighbors_amount, neighbors_amount))

    # create & init distance/neighbors matrix
    for v_id in range(neighbors_amount):
        vertex = {'x': points[v_id][1], 'y': points[v_id][2]}
        for neighbor_id in range(neighbors_amount):
            if v_id == neighbor_id:
                neighbors_matrix[v_id][neighbor_id] = np.inf
            else:
                neighbor_vertex = {'x': points[neighbor_id][1], 'y': points[neighbor_id][2]}
                x_diff = vertex['x'] - neighbor_vertex['x']
                y_diff = vertex['y'] - neighbor_vertex['y']
                distance = np.round(np.sqrt(x_diff ** 2 + y_diff ** 2))
                neighbors_matrix[v_id][neighbor_id] = neighbors_matrix[neighbor_id][v_id] = distance

    return neighbors_matrix


def check_results(route: List[int], distance_matrix: np.ndarray, declared_cost: float, method: str, vertices: int):
    cost = sum(distance_matrix[a][b] for a, b in zip(route, route[1:]))
    if cost != declared_cost:
        raise Exception(f"Invalid cost {cost} != {declared_cost} using method {method}")
    elif len(route) - 1 != vertices:
        print(len(route), route)
        raise Exception(f"Invalid vertices amount {len(route) - 1} != {vertices} using method {method}")


def visualize_route(route: List[int], points, path: str, filename: str):
    fig, ax = plt.subplots()

    x, y = [points[v_id][1] for v_id in route], [points[v_id][2] for v_id in route]

    ax.scatter(x, y)
    ax.plot(x, y)
    fig.savefig(path + '/' + filename)


def main(instances_path: str, repeat: int):
    instances_path = Path(instances_path)
    output_path = Path('output')
    check_dir(instances_path)
    check_dir(output_path)

    algorithms = [heu.nng, heu.ceg, heu.trg]

    for instance in instances_path.iterdir():
        with instance.open('r') as f_instance:
            r = reader(f_instance, delimiter=' ')
            instance_name: str = next(r)[1].split('.')[0]
            print('Instance name', instance_name)

            with open(f"output/results/{instance_name}.json", 'w') as f_write:
                points = [[int(strnum) for strnum in line] for line in r if line[0].isnumeric()]
                distance_matrix = create_distance_matrix(points)
                vertices_amount = distance_matrix.shape[0]

                json = []
                indices = [i for i in range(vertices_amount)]
                metadata = {alg.__name__: [] for alg in algorithms}

                for i in range(repeat):
                    start_vertex = np.random.choice(indices)
                    indices.remove(start_vertex)
                    print('Start vertex', start_vertex)

                    results = []
                    for alg in algorithms:
                        method_name = alg.__name__
                        cost, route = alg(distance_matrix.copy(), 0.5, start_vertex)
                        check_results(route, distance_matrix, cost, method_name, 50)

                        results.append({
                            'method': method_name,
                            'cost': cost,
                            'route': [*map(int, route)]
                        })

                        metadata[method_name].append((cost, route, points))

                    json.append(results)

                metadata = {alg.__name__: {
                    'min': min(metadata[alg.__name__]),
                    'mean': mean(x[0] for x in metadata[alg.__name__]),
                    'max': max(x[0] for x in metadata[alg.__name__])
                } for alg in algorithms}

                for alg in algorithms:
                    score, route, points = metadata[alg.__name__]['min']
                    metadata[alg.__name__]['min'] = score
                    visualize_route(route, points, './output/images', f"{instance_name}_{alg.__name__}.png")

                json_dump({'meta': metadata, 'data': json}, f_write, indent=2)


if __name__ == '__main__':
    main(argv[1], int(argv[2]))
