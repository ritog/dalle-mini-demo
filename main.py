import streamlit as st

import wandb

import subprocess

st.title("Demo for DALL-E Mini")

st.write("This is a demo for DALL-E Mini, an Open Source AI model that generates images from nothing but a text prompt")

st.text_input("What do you want to generate?", key="prompt")

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Working on it: {i+1}')
    bar.progress(i+1)

st.text_input("Enter your wandb API key (not stored):", key="wandbkey")

# subprocess.run(args=st.session_state.wandbkey, stdin=f"wandb login {st.session_state.wandbkey}", shell=True)

p1 = subprocess.run(f'wandb login {st.session_state.wandbkey}', capture_output=True, shell=True)
st.write(p1.stderr)


st.write('Here\'s your image:\n')


st.write(st.session_state.prompt)
