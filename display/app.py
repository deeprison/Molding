import streamlit as st
from constant import *
from utils import get_data


st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Molder",
    page_icon="💎"
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
    st.image(INTRO_IMG)
    """
    Made by &nbsp \
        [![kyunghoonjung](https://img.shields.io/badge/-KyunghoonJung-1C0AF7)]\
            (https://github.com/kyunghoon-jung)\
        [![Diominor](https://img.shields.io/badge/-Diominor-810AF7)]\
            (https://github.com/backgom2357)\
        [![SSinyu](https://img.shields.io/badge/-SSinyu-C80AF7)]\
            (https://github.com/SSinyu)
    """

    st.write("#")
    data, diff, size = get_data()


for _ in range(6):
    st.write("#")


c1, c2, c3, c4 = st.columns(4)
with c1: # original
    st.write("## Data")
    st.image(f"./data/{data}/png/{diff.lower()}_{size}.png", width=300)
with c2: # horizontal 
    st.write("## Horizontal")
    st.image(f"./data/{data}/png/{diff.lower()}_{size}.png", width=300)
with c3: # vertical
    st.write("## Vertical")
    st.image(f"./data/{data}/png/{diff.lower()}_{size}.png", width=300)
with c4: # trained
    st.write("## Train")
    st.image(f"./data/{data}/png/{diff.lower()}_{size}.png", width=300)
