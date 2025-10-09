from src.pipeline import DiT360Pipeline
import torch

device = torch.device("cuda:0")
pipe = DiT360Pipeline.from_pretrained("black-forest-labs/FLUX.1-dev", dtype=torch.float16).to(device)
pipe.load_lora_weights("fenghora/DiT360-Panorama-Image-Generation")

image = pipe(
    "This is a panorama. The image shows a medieval castle stands proudly on a hilltop surrounded by autumn forests, with golden light spilling across the landscape.",
    width=2048,
    height=1024,
    num_inference_steps=28,
    guidance_scale=2.8,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image.save("result.png")
