import numpy as np
import random
import operator
import os
import imageio
from tqdm import tqdm
from os.path import join
from itertools import permutations
import matplotlib.pyplot as plt
from ga_init import (
    initial_generation, custom_init_population, get_possible_coord
)


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
    return ranked, rank_info


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


def save_images(data, solution, save_dir="./saved/"):
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
        # plt.text(sol[1], sol[0], f"{i+1}", fontsize=5)
        plt.title(f"Time: {t:.2f}", fontsize=15)
        plt.savefig(join(save_dir, f"f{i+1}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()


def confirm_aircut(pc, coo):
    na = [
        (pc[0], pc[1]+1), (pc[0], pc[1]-1), (pc[0]+1, pc[1]), (pc[0]-1, pc[1]),
        (pc[0]+1, pc[1]+1), (pc[0]+1, pc[1]-1), (pc[0]-1, pc[1]+1), (pc[0]-1, pc[1]-1)
    ]
    if coo not in na:
        return 1
    return 0


def save_images_rgb(data, solution, save_dir="./saved/"):
    data3 = np.concatenate([data[..., np.newaxis] for _ in range(3)], axis=-1)

    plt.imshow(data3, vmin=0, vmax=1); plt.axis("off")
    plt.title(f"Time:{0}s    AirCut:{0}", fontsize=15)
    plt.savefig(join(save_dir, f"f0.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    px, py = solution[0][0], solution[0][1]
    data3[px, py, :] = [.9, .1, .1]
    plt.imshow(data3, vmin=0, vmax=1); plt.axis("off")
    plt.title(f"Time:{0}s    AirCut:{0}", fontsize=15)
    plt.savefig(join(save_dir, "f1.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    t, ac = 1, 0
    for ind, (x, y) in tqdm(enumerate(solution[1:]), total=len(solution)-1):
        t += get_distance((px, py), (x, y))
        ac += confirm_aircut((px, py), (x, y))

        if sum(data3[x, y, :]) == 0:
            data3[x, y, :] = [.9, .1, .1]
        else:
            data3[x, y, :] = [.0, .0, .3]
        data3[px, py, :] = [.7, .7, .7]

        px, py = x, y
        plt.imshow(data3, vmin=0, vmax=1); plt.axis("off")
        plt.title(f"Time:{t:.2f}s    AirCut:{int(ac)}", fontsize=15)
        plt.savefig(join(save_dir, f"f{ind+2}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()


def save_gif(png_dir="./saved/", duration=.1, save_path="./tmp.gif", loop=1):
    images = []
    files = sorted(os.listdir(png_dir), key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(join(png_dir, f)))
        os.remove(join(png_dir, f))

    imageio.mimsave(save_path, images, duration=duration, loop=loop)


def vis(route, data=None, save_path="./tmp.gif"):
    if data is None:
        _data = np.zeros((5, 5)).astype(np.float32)
    else:
        _data = data.copy()
    # save_images(_data, route, "./saved")
    save_images_rgb(_data, route, "./saved")
    save_gif("./saved", .4, save_path, 0)


random.seed(1991)
np.random.seed(1991)

_type = "a"
_size = "10"
data = np.load(f"./alphabet/{_type}/{_type}_{_size}.npy")
data = data.astype(np.float32)

n_population = 1000
init_type = "custom"
possible_coords, _ = get_possible_coord(data)

if init_type == "rand":
    population = initial_generation(n_population, possible_coords)
elif init_type == "mix":
    n_custom = int(n_population*0.3)
    custom_population = custom_init_population(data, n_custom)
    population = initial_generation(n_population-n_custom, possible_coords)
    population = population + custom_population
else:
    population = custom_init_population(data, n_population)

max_n_generation = 1000
for g in range(1, max_n_generation+1):
    ranked, rank_info = rank_population(population)
    parent, child = breed_population(population, ranked, n_elite=.5)
    population = mutate_population(parent + child, rate=.01)
    # print(f"{g} Generations min:{min(prev_values)} mean:{np.mean(prev_values)}")
    if g % 10 == 0:
        prev_values = list(rank_info.values())
        print(f"{g} Generations - [Min:{min(prev_values):.3f}]  [Mean:{np.mean(prev_values):.3f}]")


i = 0
frank, f_info = rank_population(population)
solution = population[frank[i]]
score = f_info[frank[i]]
print(score)

vis(solution, data)
