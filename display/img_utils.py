import os
import imageio
import matplotlib.pyplot as plt


def save_images(size_level, diff_level, images, save_path="./", readme=False):
    """
    :param size_level: ex) range(5, 21, 5)
    :param diff_level: ex) ["Low", "Medium", "High", "Extreme"]
    :param images: ex) [sq_5_low, sq_5_medium, sq_5_high, ....]
    """
    img_idx = 0
    subtitles = []
    for size in size_level:
        for diff in diff_level:
            subtitles.append(f"{diff} ({size}x{size})")

            plt.imshow(images[img_idx], plt.cm.gray)
            plt.axis("off")
            plt.savefig(
                os.path.join(save_path, f"{diff}_{size}x{size}.png"),
                bbox_inches="tight", pad_inches=0
            )
            img_idx += 1

    if readme:
        fig, ax = plt.subplots(
            len(size_level), len(diff_level), constrained_layout=True
        )
        img_idx = 0
        for s in range(len(size_level)):
            for d in range(len(diff_level)):
                ax[s][d].imshow(images[img_idx], plt.cm.gray)
                ax[s][d].set_title(subtitles[img_idx], fontsize=10)
                ax[s][d].axis("off")
                img_idx += 1
        plt.savefig(
            os.path.join(save_path, "vis.png"),
            bbox_inches="tight", pad_inches=0
        )    


def save_cont_imgs(img, mode="horizontal", save_path="./", cut_t=1, air_t=.5):
    assert mode in ["horizontal", "vertical"]
    assert img.shape[0] == img.shape[1]

    plt.imshow(img, plt.cm.gray); plt.axis("off"); plt.title(f"Time: {0:}s", fontsize=15)
    plt.savefig(os.path.join(save_path, "f0.png"), bbox_inches="tight", pad_inches=0)

    fig_num, t = 1, 0
    for i in range(img.shape[0]):
        n = 0 if i % 2 == 0 else img.shape[0]-1
        while (n >= 0) and (n < img.shape[0]):
            step = 1 if i % 2 == 0 else -1
            if mode == "horizontal":
                if img[i, n] == 0:
                    img[i, n] = .5
                    t += cut_t
                else:
                    img[i, n] = .9
                    t += air_t
            else:
                if img[n, i] == 0:
                    img[n, i] = .5
                    t += cut_t
                else:
                    img[n, i] = .9
                    t += air_t

            plt.imshow(img, plt.cm.gray, vmin=0, vmax=1); plt.axis("off"); plt.title(f"Time: {t:}s", fontsize=15)
            plt.savefig(os.path.join(save_path, f"./f{fig_num}.png"), bbox_inches="tight", pad_inches=0)
            n += step
            fig_num += 1


def get_gif(dir, save_path="./vis.gif", duration=.1):
    images = []
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(os.path.join(dir, f)))

    imageio.mimsave(save_path, images, duration=duration)


# def show_image(data, diff, size):
#     fig_path = f"./data/{data}/npy/{diff.lower()}_{size}.npy"
#     image = np.load(fig_path)
#     # st.image(image, width=300)

#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.imshow(image, plt.cm.gray)
#     ax.axis("off")
#     st.pyplot(fig)
