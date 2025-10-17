import os
import gc
import glob
import json
import torch
torch.set_float32_matmul_precision('high')
import yaml
from tqdm import tqdm
from peft import LoraConfig
from collections import OrderedDict
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
from pa_src.utils import *


device = torch.device("cuda:0")
timestep = 50
seed = 0
guidance = 2.8
dtype = torch.float16
tau = 50    # range from 0~100, the smaller the tau value, the stronger the image consistency but may reduce image quality

pipe = RFInversionParallelFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype).to(device)
pipe.load_lora_weights("fenghora/DiT360-Panorama-Image-Generation")

height = 1024
width = 2048
latent_h = height // (pipe.vae_scale_factor * 2)
latent_w = width // (pipe.vae_scale_factor * 2)
img_dims = latent_h * (latent_w + 2)

prompt = "This is a panorama image. The image depicts a village next to a snow-capped mountain"
new_prompt = "This is a panorama image. The image depicts a village next to a snow-capped mountain" # for example

init_image_path = "example/example.png"
init_image = Image.open(init_image_path).convert('RGB').resize((width, height))

mask = create_mask("example/masks/mask0.png", latent_w, latent_h).float()
# mask = 1 - mask # for inpainting
mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)

inverted_latents, image_latents, latent_image_ids = pipe.invert( 
    source_prompt="", 
    image=init_image, 
    height=height,
    width=width,
    num_inversion_steps=timestep, 
    gamma=1.0)

set_flux_transformer_attn_processor(
    pipe.transformer,
    set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
        name=name, tau=tau/100, mask=mask, device=device, img_dims=img_dims),
)

image = pipe(
    [prompt, new_prompt], 
    inverted_latents=inverted_latents,
    image_latents=image_latents,
    latent_image_ids=latent_image_ids,
    height = height,
    width = width,
    start_timestep=0.0, 
    stop_timestep=0.99,
    num_inference_steps=timestep,
    eta=1.0, 
    generator=torch.Generator(device=device).manual_seed(seed),
    mask=mask,
    use_timestep=True
).images[1]

image.save("result.png")
