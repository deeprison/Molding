import numpy as np
from copy import copy

def cal_distance(sol, data):
    distance = 0
    for i in range(1, len(sol)):
        start = sol[i-1]
        end = sol[i]
        distance += np.sqrt((data[start][0] - data[end][0])**2+(data[start][1] - data[end][1])**2)
    return distance

def ucl_distance(a,b):
    return np.sqrt((a[0] - b[0])**2+(a[1] - b[1])**2)

def generate_distance_matrix(data):
    size = len(data)
    matrix = np.zeros((size, size))
    for y in range(size):
        for x in range(size):
            matrix[y][x] = ucl_distance(data[y], data[x])
    return matrix

def swap(path, swap_first, swap_last):
        path_updated = np.concatenate((path[0:swap_first],
                                       path[swap_last:-len(path) + swap_first - 1:-1],
                                       path[swap_last + 1:len(path)]))
        return path_updated.tolist()

def two_opt(initial_route, data, improvement_threshold=0.01):
    best_route = initial_route
    best_distance = cal_distance(best_route, data)
    improvement_factor = 1
    distance_matrix = generate_distance_matrix(data)
    size = len(initial_route)
    trial_count = 0
    while improvement_factor > improvement_threshold:
        previous_best = best_distance
        for swap_first in range(1, size - 2):
            for swap_last in range(swap_first + 1, size - 1):
                before_start = best_route[swap_first - 1]
                start = best_route[swap_first]
                end = best_route[swap_last]
                after_end = best_route[swap_last+1]
                before = distance_matrix[before_start][start] + distance_matrix[end][after_end]
                after = distance_matrix[before_start][end] + distance_matrix[start][after_end]

                if after < before:
                    new_route = swap(best_route, swap_first, swap_last)
                    new_distance = cal_distance(new_route, data)
                    best_route = copy(new_route)
                    best_distance = copy(new_distance)

        improvement_factor = 1 - best_distance/previous_best
        trial_count += 1
        print(f'Processing... trial count:{trial_count}', end='\r')
    print(f'Processing... trial count:{trial_count} --- DONE!')

    return best_route, best_distance


if __name__=="__main__":
    data = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    feasible_sol = [4,0,1,3,2]

    search_size = 4

    print(two_opt(feasible_sol, data))

    # for i in range(len(feasible_sol)-search_size+1):
    #     start = feasible_sol[i]
    #     end = feasible_sol[i+search_size-1]
    #     feasible_sol = evaluate_route(start, end, feasible_sol)
    #     print('feasible_sol', feasible_sol)