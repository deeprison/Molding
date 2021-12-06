import os
import imageio
import streamlit as st
import matplotlib.pyplot as plt
from constant import *


def get_data():
    sltd = st.radio(
        "구분", options=["Basic", "Alpabet"]
    )
    st.write("#")

    if sltd == "Basic":
        data = st.selectbox(
            "데이터 선택", DATASETS, index=0, key="data"
        )

        _, c = st.columns([1, 20])
        with c:
            difficulty = st.select_slider(
                "난이도", DIFFICULTY, key="difficulty"
            )
            size = st.select_slider(
                "크기", SIZE, key="size"
            )
    else:
        data, difficulty, size = None, None, None
    return data, difficulty, size


# def show_image(data, diff, size):
#     fig_path = f"./data/{data}/npy/{diff.lower()}_{size}.npy"
#     image = np.load(fig_path)
#     # st.image(image, width=300)

#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.imshow(image, plt.cm.gray)
#     ax.axis("off")
#     st.pyplot(fig)


def save_cont_imgs(img, mode="horizontal", save_path="./"):
    assert mode in ["horizontal", "vertical"]
    assert img.shape[0] == img.shape[1]

    fig_num = 0
    for i in range(img.shape[0]):
        n = 0 if i % 2 == 0 else img.shape[0]-1
        while (n >= 0) and (n < img.shape[0]):
            step = 1 if i % 2 == 0 else -1
            if mode == "horizontal":
                if img[i, n] == 0:
                    img[i, n] = .5
                else:
                    img[i, n] = .9
            else:
                if img[n, i] == 0:
                    img[n, i] = .5
                else:
                    img[n, i] = .9

            plt.imshow(img); plt.axis("off"); plt.savefig(os.path.join(save_path, f"./f{fig_num}.png"))
            n += step
            fig_num += 1


def get_gif(dir, save_path="./"):
    images = []
    files = os.listdir(dir)
    files = sorted(files, key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(os.path.join(dir, f)))

    imageio.mimsave(os.path.join(save_path, "tmp.gif"), images, duration=.1)


# save_cont_imgs(sq_10_extreme, save_path="./tmp")
# get_gif("./tmp")
    
    