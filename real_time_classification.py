import streamlit as st
from streamlit_webrtc import webrtc_streamer

def run():
    st.title("WebRTC Test")
    webrtc_streamer(key="test")
