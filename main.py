import streamlit as st

import wandb

import subprocess

import random

import jax
import flax.linen as nn
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate

from transformers.models.bart.modeling_flax_bart import *
from transformers import BartTokenizer, FlaxBartForConditionalGeneration

import io

import requests
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


from modeling_flax_vqgan import VQModel


st.title("Demo for DALL-E Mini")

st.write("This is a demo for DALL-E Mini, an Open Source AI model that generates images from nothing but a text prompt")

api_in_flag = st.text_input("Enter your wandb API key (not stored):", key="wandbkey")


OUTPUT_VOCAB_SIZE = 16384 + 1  # encoded image token space + 1 for bos
OUTPUT_LENGTH = 256 + 1  # number of encoded tokens + 1 for bos
BOS_TOKEN_ID = 16384
BASE_MODEL = 'facebook/bart-large'

class CustomFlaxBartModule(FlaxBartModule):
    def setup(self):
        # we keep shared to easily load pre-trained weights
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        # a separate embedding is used for the decoder
        self.decoder_embed = nn.Embed(
            OUTPUT_VOCAB_SIZE,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )
        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

        # the decoder has a different config
        decoder_config = BartConfig(self.config.to_dict())
        decoder_config.max_position_embeddings = OUTPUT_LENGTH
        decoder_config.vocab_size = OUTPUT_VOCAB_SIZE
        self.decoder = FlaxBartDecoder(decoder_config, dtype=self.dtype, embed_tokens=self.decoder_embed)

class CustomFlaxBartForConditionalGenerationModule(FlaxBartForConditionalGenerationModule):
    def setup(self):
        self.model = CustomFlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            OUTPUT_VOCAB_SIZE,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.final_logits_bias = self.param("final_logits_bias", self.bias_init, (1, OUTPUT_VOCAB_SIZE))

class CustomFlaxBartForConditionalGeneration(FlaxBartForConditionalGeneration):
    module_class = CustomFlaxBartForConditionalGenerationModule



if api_in_flag:
    p1 = subprocess.run(f'wandb login {st.session_state.wandbkey}', capture_output=True, shell=True)
    st.write(p1.stderr)
    st.write('Here\'s your image:\n')

    seq = st.text_input("What do you want to generate?", key="prompt")
    
    if seq:

        run = wandb.init()
        artifact = run.use_artifact('wandb/hf-flax-dalle-mini/model-1ef8yxby:latest', type='bart_model')
        artifact_dir = artifact.download()

        # create our model and initialize it randomly
        tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)
        model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)
        model.config.force_bos_token_to_be_generated = False
        model.config.forced_bos_token_id = None
        model.config.forced_eos_token_id = None

        model.config.temperature = 4.

        vqgan = VQModel.from_pretrained("flax-community/vqgan_f16_16384")

        
        def custom_to_pil(x):
            x = np.clip(x, 0., 1.)
            x = (255*x).astype(np.uint8)
            x = Image.fromarray(x)
            if not x.mode == "RGB":
                x = x.convert("RGB")
            return x

        def generate(input, rng, params):
            return model.generate(
            **input,
            max_length=257,
            num_beams=1,
            do_sample=True,
            prng_key=rng,
            eos_token_id=50000,
            pad_token_id=50000,
            params=params
        )

        def get_images(indices, params):
            return vqgan.decode_code(indices, params=params)


        p_generate = jax.pmap(generate, "batch")
        p_get_images = jax.pmap(get_images, "batch")

        bart_params = replicate(model.params)
        vqgan_params = replicate(vqgan.params)

        prompt = [st.session_state.prompt]
        inputs = tokenizer(prompt, return_tensors='jax', padding="max_length", truncation=True, max_length=128).data
        inputs = shard(inputs)


        key = random.randint(0, 1e6) # chage this to get different output
        rngs = jax.random.PRNGKey(key)
        rngs = jax.random.split(rngs, jax.local_device_count())


        indices = p_generate(
            inputs,
            rngs,
            bart_params,
        ).sequences # takes a while to compile, after the first call, should be pretty fast
        indices = indices[:, :, 1:]

        images = p_get_images(indices, vqgan_params)
        images = np.squeeze(np.asarray(images), 1)

        img = custom_to_pil(images[0])
        st.write(img)

