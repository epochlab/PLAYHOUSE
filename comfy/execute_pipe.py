#!/usr/bin/env python3

import os, sys, warnings
sys.path.append('/mnt/vanguard/ComfyUI')
warnings.filterwarnings('ignore')

import random
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch

from main import load_extra_path_config
from nodes import init_extra_nodes, CheckpointLoaderSimple, ControlNetLoader, ControlNetApply, CLIPTextEncode, EmptyLatentImage, LoadImage, VAEDecode, VAEEncode, KSampler, NODE_CLASS_MAPPINGS

load_extra_path_config("extra_model_paths.yaml")
init_extra_nodes()

def save_image(img:Image.Image, base_filename:str, directory:str, ext:str="png") -> None:
    os.makedirs(directory, exist_ok=True)
    i = 1
    while True:
        filename = f"{base_filename}_v{i:03d}.{ext}"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            img.save(filepath)
            break
        i += 1

@dataclass
class HyperConfig:
    MODEL = "RealVisXL_V4.0.safetensors"
    cnet_depth = "diffusers_xl_depth_full.safetensors"
    cnet_canny = "diffusers_xl_canny_full.safetensors"
    upscale = "RealESRGAN_x2.pth"

    prompt = "a photograph of a red starfish, ((high resolution, high-resolution, cinematic, technicolor, film grain, analog, 70mm, 8K, IMAX, Nat Geo, DSLR))"
    negative = "worst quality, low quality, low-res, low details, cropped, blurred, defocus, bokeh, oversaturated, undersaturated, overexposed, underexposed, letterbox, aspect ratio, formatted, jpeg artefacts, draft, glitch, error, deformed, distorted, disfigured, duplicated, bad proportions"

    VERSION = "v001"
    SOURCE = f"/mnt/vanguard/STAGE/render/{VERSION}/"
    filename = f"stage_{VERSION}_"
    albedo = str(SOURCE + filename + "albedo.png")
    depth = str(SOURCE + filename + "depth.png")
    curvature = str(SOURCE + filename + "curvature.png")

    factor = 1
    w, h = 2048 // factor, 1152 // factor

    depth_stength = 1.0
    canny_stength = 0.5

    sampler = "dpmpp_sde" # "dpmpp_2m"
    scheduler = "karras"
    num_images = 1
    infer_steps = 20
    denoise = 0.75
    cfg_scale = 8.0

    enable_img2img = False
    enable_controlnet = False
    enable_upscale = True

def main():
    config = HyperConfig()

    with torch.inference_mode():
        ckpt = CheckpointLoaderSimple().load_checkpoint(ckpt_name=config.MODEL)

        image = LoadImage()
        cd = image.load_image(image=config.albedo)
        z = image.load_image(image=config.depth)
        curv = image.load_image(image=config.curvature)

        if config.enable_img2img:
            enc = VAEEncode().encode(pixels=cd[0], vae=ckpt[2])
        else:
            emptylatentimage = EmptyLatentImage()
            enc = emptylatentimage.generate(width=config.w, height=config.h, batch_size=1)

        clipencode = CLIPTextEncode()
        clipencode_prompt = clipencode.encode(text=config.prompt, clip=ckpt[1])
        clipencode_negative = clipencode.encode(text=config.negative, clip=ckpt[1])

        if config.enable_controlnet:
            controlnet = ControlNetApply()
            controlnetloader = ControlNetLoader()
            controlnetloader_depth = controlnetloader.load_controlnet(control_net_name=config.cnet_depth)
            controlnetloader_canny = controlnetloader.load_controlnet(control_net_name=config.cnet_canny)

            clip_depth = controlnet.apply_controlnet(strength=config.depth_stength, conditioning=clipencode_prompt[0], control_net=controlnetloader_depth[0], image=z[0])
            clip_canny = controlnet.apply_controlnet(strength=config.canny_stength, conditioning=clip_depth[0], control_net=controlnetloader_canny[0], image=curv[0])
            clip = clip_canny[0]
        else:
            clip = clipencode_prompt[0]

        if config.enable_upscale:
            upscaler = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]().load_model(model_name=config.upscale)
            print("Upsampling: Enabled")

        for _ in range(config.num_images):
            seed = random.randint(1, 2**64)
            print(f"Seed: {seed}")

            latents = KSampler().sample(
                seed=seed,
                steps=config.infer_steps,
                cfg=config.cfg_scale,
                sampler_name=config.sampler,
                scheduler=config.scheduler,
                denoise=config.denoise,
                model=ckpt[0],
                positive=clip,
                negative=clipencode_negative[0],
                latent_image=enc[0],
            )

            dec = VAEDecode().decode(samples=latents[0], vae=ckpt[2])

            if config.enable_upscale:
                dec = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]().upscale(upscale_model=upscaler[0], image=dec[0])

            out = dec[0].numpy().squeeze()
            out = Image.fromarray((out * 255.0).astype(np.uint8))
            save_image(out, "output", config.SOURCE + "diffusion")

if __name__ == "__main__":
    main()