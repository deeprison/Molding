import os
from os.path import join
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_distance(crd1, crd2):
    return np.sqrt((crd1[0] - crd2[0]) ** 2 + (crd1[1] - crd2[1]) ** 2)


def confirm_aircut(pc, coo):
    na = [
        (pc[0], pc[1]+1), (pc[0], pc[1]-1), (pc[0]+1, pc[1]), (pc[0]-1, pc[1]),
        (pc[0]+1, pc[1]+1), (pc[0]+1, pc[1]-1), (pc[0]-1, pc[1]+1), (pc[0]-1, pc[1]-1)
    ]
    if coo not in na:
        return 1
    return 0


def save_images_rgb(data, solution, save_dir="./saved"):
    data3 = np.concatenate([data[..., np.newaxis] for _ in range(3)], axis=-1)

    plt.imshow(data3, vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Time:0s    AirCut:0", fontsize=15)
    plt.savefig(join(save_dir, "f0.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    px, py = solution[0][0], solution[0][1]
    data3[px, py, :] = [.9, .1, .1]
    plt.imshow(data3, vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Time:0s    AirCut:0", fontsize=15)
    plt.savefig(join(save_dir, "f1.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    t, ac = 1, 0
    for ind, (x, y) in tqdm(enumerate(solution[1:]), total=len(solution)-1):
        t += get_distance((px, py), (x, y))
        ac += confirm_aircut((px, py), (x, y))

        if sum(data3[x, y, :]) == 0:
            data3[x, y, :] = [.9, .1, .1]
        else:
            data3[x, y, :] = [.9, .1, .1]  # [.0, .0, .3]
        data3[px, py, :] = [.7, .7, .7]

        px, py = x, y
        plt.imshow(data3, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Time:{t:.2f}s    AirCut:{int(ac)}", fontsize=15)
        plt.savefig(join(save_dir, f"f{ind+2}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()


def save_gif(png_dir="./saved", duration=.1, save_path="./tmp.gif", loop=0):
    images = []
    files = sorted(os.listdir(png_dir), key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(join(png_dir, f)))
        os.remove(join(png_dir, f))

    imageio.mimsave(save_path, images, duration=duration, loop=loop)
