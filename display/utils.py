import streamlit as st
from constant import *


def get_data_info():
    sltd = st.radio(
        "구분", options=["Basic", "Alpabet"]
    )
    st.write("#")

    if sltd == "Basic":
        data = st.selectbox(
            "데이터 선택", DATASETS_BASIC, index=0, key="data_basic"
        )

        _, c = st.columns([1, 20])
        with c:
            difficulty = st.select_slider(
                "난이도", DIFFICULTY_BASIC, key="difficulty_basic"
            )
            size = st.select_slider(
                "크기", SIZE_BASIC, key="size_basic"
            )
    else:
        data = st.selectbox(
            "데이터 선택", DATASETS_ALPABETS, index=0, key="data_alphabet"
        )
        _, c = st.columns([1, 20])
        with c:
            size = st.select_slider(
                "크기", SIZE_ALPABETS, key="size_alphabet"
            )
        difficulty = None

    return data, difficulty, size


def visualize_basic(data, diff, size, width=200):
    st.write("## Data")
    st.image(f"./data/{data}/png/{diff.lower()}_{size}.png", width=width)
    st.write("#")

    c1, c2, c3 = st.columns(3)
    with c1: 
        st.write("#### Baby-level")
        st.image(BASIC_GIF[f"{diff.lower()}_{size}"], width=width)
    with c2:
        st.write("#### Human-level")
        st.image(SOLUTION_GIF[f"{diff.lower()}_{size}"], width=width)
    with c3:
        st.write("#### Trained")
        st.image(SOLUTION_GIF[f"{diff.lower()}_{size}"], width=width)


def visualize_alphabet(data, size, width=200):
    st.write("## Data")
    # st.image()
    st.write("#")

    c1, c2, c3 = st.columns(3)
    with c1:
        # horizontal trajectory
        st.write("#### ")
    with c2:
        # human-level trajectory
        st.write("#### ")
    with c3:
        # trained model trajectory
        st.wrtie("#### ")
    
    