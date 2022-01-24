from itertools import permutations
from initializer import *
from ga_utils import *


class Solver:
    def __init__(self, env, n_routes, n_generations, elite_ratio=.5, mutate_ratio=.01, print_iter=10):
        self.Initializer = InitRoute(env, n_routes)
        self.n_routes = n_routes
        self.n_generations = n_generations
        self.elite_ratio = elite_ratio
        self.mutate_ratio = mutate_ratio
        self.print_iter = print_iter

        self.init_routes = self.Initializer.routes

    def train(self):
        routes, _ = self.build_next_generation(self.init_routes)
        for i in range(1, self.n_generations):
            routes, rank_info = self.build_next_generation(routes)
            if i % self.print_iter == 0:
                rank_info = list(rank_info.values())
                print(f"{i} iterations - [Min:{min(rank_info):.3f}] [Mean:{np.mean(rank_info):.3f}]")

        self.routes = routes

    def build_next_generation(self, routes):
        ranked, rank_info = rank_routes(routes)
        parent, child = self.breed_routes(routes, ranked, self.elite_ratio)
        new_routes = self.mutate_routes(parent+child, self.mutate_ratio)
        return new_routes, rank_info

    def get_routes(self, idx):
        ranked, rank_info = rank_routes(self.routes)
        solution = self.routes[ranked[idx]]
        score = rank_info[ranked[idx]]
        return solution, score

    @staticmethod
    def breed_routes(routes, ranked, n_elite=None):
        if n_elite is None:
            n_elite = len(ranked) // 2
        if isinstance(n_elite, float):
            n_elite = int(len(ranked) * n_elite)

        pool_indexes = ranked[:n_elite]
        parents = [routes[i] for i in pool_indexes]
        parents_perm = list(permutations(pool_indexes, 2))
        parents_perm = random.sample(parents_perm, n_elite)

        child = []
        for prai, prbi in parents_perm:
            child.append(breed(routes[prai], routes[prbi]))
        return parents, child

    @staticmethod
    def mutate_routes(routes, ratio=.01):
        mutated = []
        for route in routes:
            if np.random.random() < ratio:
                mutated.append(mutate(route))
            else:
                mutated.append(route)
        return mutated
