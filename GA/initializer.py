import random
import numpy as np


def get_distance(crd1, crd2):
    return np.sqrt((crd1[0] - crd2[0]) ** 2 + (crd1[1] - crd2[1]) ** 2)


def get_crds(env):
    pos_crds, imp_crds = [], []
    for i in range(len(env)):
        for j in range(len(env[0])):
            if env[i, j] == 0:
                pos_crds.append((i, j))
            else:
                imp_crds.append((i, j))
    return pos_crds, imp_crds


def get_candidates(env, crd, ways=None, died=None):
    if ways is None:
        ways = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    if died is None:
        died = []

    candidates = []
    for i, j in ways:
        x, y = crd[0]+i, crd[1]+j
        if (0 <= x < env.shape[0]) and (0 <= y < env.shape[1]) \
                and (env[x, y] == 0) and (x, y) not in died:
            candidates.append((x, y))
    return candidates


def random_cont(env, crd=None, lin=None, died=None):
    pos_crds, imp_crds = get_crds(env)
    if crd is None:
        crd = random.sample(pos_crds, 1)[0]
    elif crd in imp_crds:
        crd = random.sample(get_candidates(env, crd, died), 1)[0]

    if np.random.random() < .5:
        ways = [(0, 1), (0, -1)] if lin else None
    else:
        ways = [(1, 0), (-1, 0)] if lin else None

    if died is None:
        died_crds = [crd]
    else:
        died_crds = died + [crd]

    while len(died_crds) < len(pos_crds):
        candidates = get_candidates(env, crd, ways, died_crds)
        while len(candidates) > 0:
            crd = random.sample(candidates, 1)[0]
            died_crds.append(crd)
            candidates = get_candidates(env, crd, ways, died_crds)

        if len(died_crds) == len(pos_crds):
            break
        remains = list(set(pos_crds) - set(died_crds))
        remains_distance = [get_distance(crd, rem) for rem in remains]
        crd = remains[np.argmin(remains_distance)]
        died_crds.append(crd)
    return died_crds


class InitRoute:
    def __init__(self, env, n_routes="auto"):
        self.env = env
        self.pos_crds, self.imp_crds = get_crds(env)

        if n_routes == "auto":
            lin_routes = self.lin_generation()
            cont_routes = []
            for crd in self.pos_crds:
                route = self.random_cont_generation(5, crd, True)
                cont_routes.extend(route)
            # cont_routes = self.random_cont_generation(n_routes - len(lin_routes))
            self.routes = lin_routes + cont_routes
            print(f"Use {len(self.routes)} routes")

    def random_generation(self, n_routes):
        routes = []
        for _ in range(n_routes):
            _route = random.sample(self.pos_crds, len(self.pos_crds))
            routes.append(_route)
        return routes

    def remove_crds(self, routes):
        return [crd for crd in routes if crd not in self.imp_crds]

    def lin_generation(self):
        n = self.env.shape[0]
        routes = [
            [(i, j) for i in range(n) for j in range(n)],  # >v>
            [(j, i) for i in range(n) for j in range(n)],  # v>v
            [(i, j) for i in range(n-1, -1, -1) for j in range(n)],  # >^>
            [(j, i) for i in range(n-1, -1, -1) for j in range(n)],  # v<v
            [(i, j) for i in range(n) for j in range(n-1, -1, -1)],  # <v<
            [(j, i) for i in range(n) for j in range(n-1, -1, -1)],  # ^>^
            [(i, j) for i in range(n-1, -1, -1) for j in range(n-1, -1, -1)],  # <^<
            [(j, i) for i in range(n - 1, -1, -1) for j in range(n - 1, -1, -1)]  # ^<^
        ]

        _route = [[], [], [], []]  # >v< , v>^ , >^< , v<^
        for i in range(n):
            j = 0 if i % 2 == 0 else n-1
            while (j >= 0) and (j < n):
                step = 1 if i % 2 == 0 else -1
                _route[0].append((i, j))
                _route[1].append((j, i))
                j += step

        for i in range(n-1, -1, -1):
            j = 0 if i % 2 == 0 else n-1
            while (j >= 0) and (j < n):
                step = 1 if i % 2 == 0 else -1
                _route[2].append((i, j))
                _route[3].append((j, i))
                j += step
        routes.extend(_route)

        routes = [self.remove_crds(r) for r in routes]
        return routes

    def random_cont_generation(self, n_routes, crd=None, lin=None):
        routes = []
        for _ in range(n_routes):
            if lin is None:
                lin = True if np.random.random() < .5 else None
            routes.append(random_cont(self.env, crd, lin))
        return routes

    @staticmethod
    def breed(pg1, pg2):
        cg1, cg2 = [], []

        cg_part1 = int(random.random() * len(pg1))
        cg_part2 = int(random.random() * len(pg2))
        prg = range(min(cg_part1, cg_part2), max(cg_part1, cg_part2))
        for i in prg:
            cg1.append(pg1[i])

        cg2 = [i for i in pg2 if i not in cg1]
        return cg1 + cg2
