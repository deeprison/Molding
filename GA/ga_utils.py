import random
import numpy as np
from operator import itemgetter
from initializer import get_distance


def calc_distance_sum(route):
    distance = 0
    for idx in range(len(route) - 1):
        distance += get_distance(route[idx], route[idx+1])
    return distance


def rank_routes(routes):
    rank_info = {i: calc_distance_sum(routes[i]) for i in range(len(routes))}
    ranked = sorted(rank_info.items(), key=itemgetter(1))
    ranked = [i[0] for i in ranked]
    return ranked, rank_info


def get_cix(route):
    cix = []
    px, py = route[0]
    for i, (x, y) in enumerate(route[1:]):
        if (abs(px-x) > 1) or (abs(py-y) > 1):
            cix.append(i+1)
        px, py = x, y
    return cix


def breed(pra, prb, cix_ratio=.0):
    cra, crb = [], []
    if np.random.random() < cix_ratio:
        cix = get_cix(pra)
        route_a_idx = np.random.choice(range(len(cix)))
        route_b_idx = np.random.choice(range(len(cix[route_a_idx:]))) + route_a_idx
        starts, ends = cix[route_a_idx], cix[route_b_idx]
    else:
        route_a = int(random.random() * len(pra))
        route_b = int(random.random() * len(prb))
        starts, ends = min(route_a, route_b), max(route_a, route_b)

    for i in range(starts, ends):
        cra.append(pra[i])

    crb = [i for i in prb if i not in cra]
    return cra + crb


def mutate(route):
    swap_a = int(random.random() * len(route))
    swap_b = int(random.random() * len(route))

    sa, sb = route[swap_a], route[swap_b]
    route[swap_a], route[swap_b] = sb, sa
    return route
