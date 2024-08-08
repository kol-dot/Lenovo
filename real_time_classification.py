import streamlit as st
from streamlit_webrtc import webrtc_streamer

def run():
    st.title("Real-Time Staff Classification")
    webrtc_streamer(key="example")
