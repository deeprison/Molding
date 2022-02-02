import numpy as np
from solver import Solver
from vis import *


MODE = "a"
SIZE = 10
ENV = np.load(f"./alphabet/{MODE}/{MODE}_{SIZE}.npy").astype(np.float32)
N_ROUTES = "auto"
N_GENE = 100
ELITE_RATIO = .5
MUTATE_RATIO = .01
PRINT_ITER = 1

solver = Solver(ENV, N_ROUTES, N_GENE, ELITE_RATIO, MUTATE_RATIO, PRINT_ITER)
solver.train()
solution, score = solver.get_routes(0)

init_routes = solver.init_routes


data = ENV.copy()
save_images_rgb(data, solution)
save_gif(save_path=f"{MODE}{SIZE}.gif")
