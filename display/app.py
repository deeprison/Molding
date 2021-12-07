import time
import streamlit as st
from constant import *
from utils import get_data_info, visualize


st.set_page_config(
    layout="centered", # wide
    initial_sidebar_state="auto",
    page_title="Molder",
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
    data, diff, size = get_data_info()


for _ in range(1):
    st.write("#")


visualize(data, diff, size, 200)