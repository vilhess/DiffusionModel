import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from DDPM import ContextDDPM
from UNet import ContextUNET
from utils import int2class, class2int

# Set up device
DEVICE = "mps"

# Title and description
st.title('Conditional Diffusion Model')
st.markdown("""
This app generates images using a conditional diffusion model. Select a class and view how the generated image evolves over time.
""")

# Load model checkpoint
try:
    checkpoints = torch.load('checkpoints/best_model_mnist_context.pkl')
except FileNotFoundError:
    st.error("Model checkpoint not found. Please ensure the path is correct.")
    st.stop()

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = ContextDDPM(ContextUNET(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)
ddpm.load_state_dict(checkpoints["model"])

# Select class
classes = list(class2int.keys())
classe = st.selectbox("Select the class you want to generate", classes)
context = class2int[classe]

# Generate Image Section
st.subheader(f"Generating {classe}")

if st.button('Compute Image'):
    with st.spinner('Generating image...'):
        ddpm.save_streamlit(context=context)
        st.success('Image generated!')

# Slider for Timestep
st.markdown("### Explore Image Generation Progress")
timestep = st.slider('Select the timestep', min_value=0, max_value=999)

# Load and display image
try:
    img = np.load(f"streamlit_images/img_{999-timestep}.npy")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, cmap="gray")
    ax.axis('off')
    st.pyplot(fig)
except FileNotFoundError:
    st.error("Generated image not found. Please generate the image first.")

# Display GIF of entire generation process
st.markdown("### Full Generation Process -- GIF")
st.image('ddpm.gif', caption="Image Generation Over Time", width=700)  # Adjust the width to your preference

