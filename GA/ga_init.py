import os
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join


def remove_coords(individual, imp_coords):
    p_coords = []
    for crd in individual:
        if crd not in imp_coords:
            p_coords.append(crd)
    return p_coords


def get_possible_coord(ary):
    coords, exc = [], []
    for i in range(len(ary)):
        for j in range(len(ary[0])):
            if ary[i, j] == 0:
                coords.append((i, j))
            else:
                exc.append((i, j))
    return coords, exc


def get_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def initial_generation(population_size, coords):
    def random_generation():
        route = random.sample(coords, len(coords))
        return route

    population = []
    for i in range(0, population_size):
        population.append(random_generation())
    return population


def custom_init_population(data, n_breed):
    n = data.shape[0]
    _, imp_coords = get_possible_coord(data)

    init = []
    init.append([(i, j) for i in range(n) for j in range(n)])
    init.append([(j, i) for i in range(n) for j in range(n)])
    init.append([(i, j) for i in range(n - 1, -1, -1) for j in range(n)])
    init.append([(j, i) for i in range(n - 1, -1, -1) for j in range(n)])
    init.append([(i, j) for i in range(n) for j in range(n - 1, -1, -1)])
    init.append([(j, i) for i in range(n) for j in range(n - 1, -1, -1)])
    init.append([(i, j) for i in range(n - 1, -1, -1) for j in range(n - 1, -1, -1)])
    init.append([(j, i) for i in range(n - 1, -1, -1) for j in range(n - 1, -1, -1)])

    _init = [[], []]
    for i in range(n):
        j = 0 if i % 2 == 0 else n - 1
        while (j >= 0) and (j < n):
            step = 1 if i % 2 == 0 else -1
            _init[0].append((i, j))
            _init[1].append((j, i))
            j += step
    init.extend(_init)

    _init = [[], []]
    for i in range(n - 1, -1, -1):
        j = 0 if i % 2 == 0 else n - 1
        while (j >= 0) and (j < n):
            step = 1 if i % 2 == 0 else -1
            _init[0].append((i, j))
            _init[1].append((j, i))
            j += step
    init.extend(_init)

    _init = [[], []]
    for i in range(n):
        j = 0
        while i >= 0:
            _init[0].append((i, j))
            _init[1].append((j, i))
            j += 1
            i -= 1
    for j in range(1, n):
        i = n - 1
        while j <= (n - 1):
            _init[0].append((i, j))
            _init[1].append((j, i))
            i -= 1
            j += 1
    init.extend(_init)

    _init = [[], []]
    for i in range(n):
        j = n - 1
        while (j >= 0) and (i >= 0):
            _init[0].append((i, j))
            _init[1].append((j, i))
            j -= 1
            i -= 1
    for j in range(n - 2, -1, -1):
        i = n - 1
        while j >= 0:
            _init[0].append((i, j))
            _init[1].append((j, i))
            i -= 1
            j -= 1
    init.extend(_init)

    _init = [[], []]
    for j in range(n - 1, -1, -1):
        i = n - 1
        while j <= (n - 1):
            _init[0].append((i, j))
            _init[1].append((j, i))
            i -= 1
            j += 1
    for i in range(n - 2, -1, -1):
        j = 0
        while i >= 0:
            _init[0].append((i, j))
            _init[1].append((j, i))
            i -= 1
            j += 1
    init.extend(_init)

    n = data.shape[0]
    _init = [[], []]
    p = 0
    while n > 0:
        i = 0
        for j in range(n):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for i in range(1, n):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for j in range(n - 2, -1, -1):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for i in range(n - 2, 0, -1):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        p += 1
        n -= 2
    init.extend(_init)

    n = data.shape[0]
    _init_t_1 = [(abs(i-(n-1)), abs(j-(n-1))) for i, j in _init[0]]
    _init_t_2 = [(abs(i-(n-1)), abs(j-(n-1))) for i, j in _init[1]]
    init.extend([_init_t_1, _init_t_2])

    n = data.shape[0]
    _init = [[], []]
    p = 0
    while n > 0:
        i = 0
        for j in range(n - 1, -1, -1):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for i in range(1, n):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for j in range(1, n):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        for i in range(n - 2, 0, -1):
            _init[0].append((i + p, j + p))
            _init[1].append((j + p, i + p))
        p += 1
        n -= 2
    init.extend(_init)

    n = data.shape[0]
    _init_t_1 = [(abs(i-(n-1)), abs(j-(n-1))) for i, j in _init[0]]
    _init_t_2 = [(abs(i-(n-1)), abs(j-(n-1))) for i, j in _init[1]]
    init.extend([_init_t_1, _init_t_2])

    init = [remove_coords(i, imp_coords) for i in init]
    indexes = len(init)
    for _ in range(n_breed - indexes):
        ind1, ind2 = np.random.choice(range(indexes), 2)
        init.append(breed(init[ind1], init[ind2]))

    return init


def breed(parent1, parent2):
    child_part1, child_part2 = [], []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent2))
    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)
    for i in range(start_gene, end_gene):
        child_part1.append(parent1[i])

    child_part2 = [i for i in parent2 if i not in child_part1]
    return child_part1 + child_part2
