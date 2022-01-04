import os
import random
import operator
from os.path import join
from shutil import copyfile
from itertools import permutations

import yaml
import wandb
import imageio
import numpy as np
import matplotlib.pyplot as plt
from box import Box
from tqdm import tqdm

    
def get_possible_coord(ary):
    coords, exc = [], []
    for i in range(len(ary)):
        for j in range(len(ary[0])):
            if ary[i, j] == 0:
                coords.append((i, j))
            else:
                exc.append((i, j))
    return coords, exc


def initial_generation(population_size, coords):
    def random_generation():
        route = random.sample(coords, len(coords))
        return route

    population = []
    for _ in range(0, population_size):
        population.append(random_generation())
    return population


def get_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def calc_distance_sum(individual):
    distance = 0
    for start_idx in range(len(individual) - 1):
        end_idx = start_idx + 1

        # dist = np.linalg.norm(individual[start_idx] - individual[end_idx])
        dist = get_distance(individual[start_idx], individual[end_idx])
        distance += dist
    return distance


def rank_population(population):
    rank_info = {}
    for i in range(0, len(population)):
        rank_info[i] = calc_distance_sum(population[i])

    ranked = sorted(rank_info.items(), key=operator.itemgetter(1))
    ranked = [i[0] for i in ranked]
    return ranked, list(rank_info.values())


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


def breed_population(population, ranked, n_elite=None):
    if n_elite is None:
        n_elite = len(ranked) // 2
    if isinstance(n_elite, float):
        n_elite = int(len(ranked) * n_elite)

    # pool_idx = [i for i in ranked[:n_elite]]
    pool_idx = ranked[:n_elite]
    parent = [population[i] for i in pool_idx]
    parent_perm = list(permutations(pool_idx, 2))
    parent_perm = random.sample(parent_perm, n_elite)

    child = []
    for p1, p2 in parent_perm:
        child.append(breed(population[p1], population[p2]))
    return parent, child


def mutate(individual, rate=.01):
    for i in range(len(individual)):
        if random.random() < rate:
            swap = int(random.random() * len(individual))
            s1, s2 = individual[i], individual[swap]
            individual[i], individual[swap] = s2, s1
    return individual


def mutate_population(population, rate=.01):
    mutated = []
    for i in range(len(population)):
        if i < int(len(population) * .001):
            mutated_individual = population[i]
        else:
            mutated_individual = mutate(population[i], rate)
        mutated.append(mutated_individual)
    return mutated


def make_next_generation(population, n_elite=.5, mutate_rate=.01):
    ranked, prev_values = rank_population(population)
    parent, child = breed_population(population, ranked, n_elite)
    next_population = mutate_population(parent + child, mutate_rate)
    return next_population, prev_values


def save_images(data, solution, save_dir="./tmp/saved/"):
    plt.imshow(data, plt.cm.gray, vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"Time: {0}s", fontsize=15)
    plt.savefig(join(save_dir, f"f{0}.png"), bbox_inches="tight", pad_inches=0)

    t = 0
    prev_x, prev_y = solution[0][0], solution[0][1]
    for i, sol in tqdm(enumerate(solution), total=len(solution)):
        t += get_distance((prev_x, prev_y), sol)

        if data[sol[0], sol[1]] == 0:
            data[sol[0], sol[1]] = .5
        else:
            data[sol[0], sol[1]] = .9

        prev_x, prev_y = sol

        plt.imshow(data, plt.cm.gray, vmin=0, vmax=1)
        plt.axis("off")
        plt.text(sol[1], sol[0], f"{i+1}", fontsize=5)
        # plt.title(f"Time: {t:.2f}", fontsize=15)
        plt.savefig(join(save_dir, f"f{i+1}.png"), bbox_inches="tight", pad_inches=0)

        plt.close()


def save_gif(png_dir="./tmp/saved/", duration=.1, save_path="./tmp/tmp.gif"):
    images = []
    files = sorted(os.listdir(png_dir), key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(join(png_dir, f)))
        os.remove(join(png_dir, f))

    imageio.mimsave(save_path, images, duration=duration, loop=1)


def remove_coords(individual, imp_coords):
    p_coords = []
    for crd in individual:
        if crd not in imp_coords:
            p_coords.append(crd)
    return p_coords


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


if __name__ == "__main__":
    with open("./config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = Box(config) 
    
    random.seed(config.seed)
    np.random.seed(config.seed)

    alphabet = config.data.alphabet
    size = config.data.size
    
    data = np.load(
        f"./alphabet/{alphabet}/{alphabet}_{size}.npy"
    )
    data = data.astype(np.float32)

    save_dir = f"./saved/{alphabet}-{size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Make {save_dir} directory")

    copyfile("./config.yml", join(save_dir, "config.yml"))

    # Initialize
    possible_coords, _ = get_possible_coord(data)
    if config.opset.rand_init is True:
        population = initial_generation(
            config.opset.n_population, possible_coords
        )
    elif isinstance(config.opset.rand_init, float):
        assert config.opset.n_population % 2 == 0
        rand_population = initial_generation(
            config.opset.n_population//2, possible_coords
        )
        custom_population = custom_init_population(
            data, config.opset.n_population//2
        )
        population = rand_population + custom_population
    else:
        population = custom_init_population(
            data, config.opset.n_population
        )

    # logging
    p_unit = config.opset.n_population/1000
    log_name = f"{alphabet.upper()}{size}-{p_unit}kp"
    if config.opset.rand_init:
        log_name += "-Rand"

    wandb.init(
        project="GA",
        entity="molding_rl",
        name=log_name,
        config=dict(config)
    )

    # Optimize
    for g in range(1, config.opset.n_generation+1):
        ranked, prev_values = rank_population(population)
        parent, child = breed_population(
            population, ranked, n_elite=config.opset.elite_ratio
        )
        population = mutate_population(
            parent+child, rate=config.opset.mutate_rate
        )
        wandb.log({
            "min": min(prev_values),
            "mean": np.mean(prev_values)
        })
        print(f"{g} Generations min:{min(prev_values):.4f} mean:{np.mean(prev_values):.4f}")

    frank, fvalue = rank_population(population)
    solution = population[frank[0]]
    np.save(join(save_dir, f"solution-{log_name}.npy"), np.array(solution))
    
   
    # save_images(data, solution, save_dir)
    # save_gif(
    #     "./tmp/saved/", 
    #     config.vis.gif_duration, 
    #     f"./tmp/solution_{alphabet}_{size}.gif"
    # )
