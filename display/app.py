import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from constants import *
from solver import Solver


def get_random_env(n, seed):
    np.random.seed(seed)

    assert n in [10, 20]
    env = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            env[i, j] = 1 if np.random.random() < .5 else 0
    env[0, :], env[:, 0] = 0, 0
    env[-1, :], env[:, -1] = 0, 0
    return env


def get_data_info(seed):
    st.write("# ➡️Data Setup")
    sltd = st.radio("Type", options=["Alphabet", "Random Shapes", "Upload"])

    if sltd == "Alphabet":
        data = st.selectbox("Data", ALPHABETS, index=0, key="alphabet")
        _, c, _ = st.columns([1, 20, 1])
        with c:
            size = st.radio("Size", SIZES, key="alphabet_size")
        env = np.array(PD_ENV[f"{data.lower()}{size.split('x')[0]}"]).astype(np.uint8)

    elif sltd == "Random Shapes":
        _, c, _ = st.columns([1, 20, 1])
        with c:
            size = st.radio("Size", SIZES, key="random_size")
        env = get_random_env(int(size.split("x")[0]), seed)

    else:
        data = st.file_uploader("File", type="csv", key="uploaded_data")
        env = pd.read_csv(data, header=None).values
        size = f"{env.shape[0]}x{env.shape[1]}"

    return env


def get_train_info():
    st.write("# ➡️Training Setup")
    method = st.selectbox("Algorithm", options=["Genetic Algorithm"], key="mehtod")
    c1, c2 = st.columns(2)
    with c1:
        n_routes = st.number_input(
            "init routes", value=1000, key="n_routes",
            min_value=1, max_value=10000
        )
    with c2:
        n_gene = st.number_input(
            "iterations", value=10, key="n_gene",
            min_value=1, max_value=10000
        )
    return n_routes, n_gene


def prep(data, crd=None, prev_crd=None):
    if data.shape[-1] != 3:
        data = np.concatenate([data[..., np.newaxis] for _ in range(3)], axis=-1)

    if crd is None:
        return data
    else:
        if prev_crd is not None:
            data[prev_crd[0], prev_crd[1], :] = [.7, .7, .7]
        data[crd[0], crd[1], :] = [.9, .1, .1]
        return data


@st.cache
def get_images(data, route):
    images = []
    _data = prep(data, route[0])
    images.append(_data)
    for crd, prev in zip(route[1:], route[:-1]):
        _data = prep(_data, crd, prev)
        images.append(_data.copy())
    return images


st.set_page_config(
    layout="centered",  # wide
    initial_sidebar_state="auto",
    page_title="CAM",
    page_icon="💎",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: %ipx;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: %ipx;
        margin-left: -%ipx;
    }
    """ % (SIDE_WIDTH, SIDE_WIDTH, SIDE_WIDTH),
    unsafe_allow_html=True,
)

with st.sidebar:
    # st.image(INTRO_IMG)
    """
    👨🏻‍💻 Made by &nbsp \
        [![kyunghoonjung](https://img.shields.io/badge/-KyunghoonJung-1C0AF7)]\
            (https://github.com/kyunghoon-jung)\
        [![Diominor](https://img.shields.io/badge/-Diominor-810AF7)]\
            (https://github.com/backgom2357)\
        [![SSinyu](https://img.shields.io/badge/-SSinyu-C80AF7)]\
            (https://github.com/SSinyu)
    """

    st.write("#")
    seed = int(st.number_input("🔌 Seed", value=2022, min_value=1, max_value=10000))
    env = get_data_info(seed)

    # n_routes, n_gene = get_train_info()
    st.write("#")
    n_gene = st.number_input(
            "Iteration", value=10, key="n_gene",
            min_value=1, max_value=10000
        )

st.header("Computer-Aided Manufacturing")
st.write("#")
st.write("#")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Dataset")
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(env, plt.cm.gray)
    plt.axis("off")
    st.pyplot(fig)

with c2:
    st.subheader("Training")
    st.write(" ")
    train_start = st.button("Training Start", key="train_start")

    sol = None
    if train_start:
        with st.spinner(text="Training..."):
            random.seed(seed)
            np.random.seed(seed)

            solver = Solver(env, "auto", n_gene, .5, .01, 1)
            routes, rank_info = solver.build_next_generation(solver.init_routes)
            rank_info = list(rank_info.values())

            _min, _mean = [], []
            for i in range(1, solver.n_generations):
                routes, rank_info = solver.build_next_generation(routes)
                rank_info = list(rank_info.values())
                _min.append(min(rank_info))
                _mean.append(np.mean(rank_info))

            solver.routes = routes
            sol, _ = solver.get_routes(0)

        fig, ax = plt.subplots()
        ax.plot(_min, label="min")
        ax.plot(_mean, label="mean")
        ax.legend()
        st.pyplot(fig)

st.write("#")
if sol is not None:
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Best Route")
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(env, plt.cm.gray)
        plt.axis("off")

        xs, ys = [], []
        for x, y in sol:
            xs.append(y)
            ys.append(x)
        ax.plot(xs, ys, "r")
        st.pyplot(fig)

    with c4:
        st.subheader("Download")
        st.write(" ")
        b_rte = pd.DataFrame(sol).to_csv(index=False, header=None).encode("utf-8")
        dl_rte = st.download_button("Route Download", b_rte, file_name="route.csv")

        np.save("./tmp.npy", env)
        _env = np.load("./tmp.npy")
        st.write(_env.shape)
