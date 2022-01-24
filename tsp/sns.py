import numpy as np
from itertools import permutations

def cal_distance(sol, data):
    distance = 0
    for i in range(1, len(sol)):
        start = sol[i-1]
        end = sol[i]
        distance += np.sqrt((data[start][0] - data[end][0])**2+(data[start][1] - data[end][1])**2)
    return distance

def destroy_route(start, end, sol):
    return sol[sol.index(start): sol.index(end)+1]

def generate_candidate_sol(candi, random=None):
    if random is not None:
        permut_list = []
        for _ in range(random):
            permut_list.append(np.random.permutation(candi))
        return permut_list
    else:
        return list(permutations(candi))

def evaluate_route(start, end, sol, data, random):
    destoried_route = destroy_route(start, end, sol)
    candidate_routes = generate_candidate_sol(destoried_route, random)
    candidate_routes = [sol[:sol.index(start)]+list(route)+sol[sol.index(end)+1:] for route in candidate_routes]
    shortest_route = 0
    shortest_distance = 999999999
    for idx, sol in enumerate(candidate_routes):
        distance = cal_distance(sol, data)
        if distance <= shortest_distance:
            shortest_route = idx
            shortest_distance = distance
    return candidate_routes[shortest_route]

def small_neighborhood_search(data, feasible_sol, search_size, random):
    min_length = cal_distance(feasible_sol, data)
    for i in range(len(feasible_sol)-search_size+1):
        print(f'processing...{i}/{len(feasible_sol)-search_size+1} - {i/(len(feasible_sol)-search_size+1)*100:.1f}%, Current_length: {cal_distance(feasible_sol, data)}', end='\r')
        start = feasible_sol[i]
        end = feasible_sol[i+search_size-1]
        current_sol = evaluate_route(start, end, feasible_sol, data, random)
        current_distance = cal_distance(current_sol, data)
        if current_distance < min_length:
            feasible_sol = current_sol
            min_length = current_distance
    print(f'processing...{i+1}/{len(feasible_sol)-search_size+1} - {(i+1)/(len(feasible_sol)-search_size+1)*100:.1f}%, Current_length: {cal_distance(feasible_sol, data)}')
    return feasible_sol


if __name__=="__main__":
    data = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    feasible_sol = [4,0,1,3,2]

    search_size = 4

    print(small_neighborhood_search(data, feasible_sol, search_size))

    # for i in range(len(feasible_sol)-search_size+1):
    #     start = feasible_sol[i]
    #     end = feasible_sol[i+search_size-1]
    #     feasible_sol = evaluate_route(start, end, feasible_sol)
    #     print('feasible_sol', feasible_sol)