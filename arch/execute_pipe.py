#!/usr/bin/env python3

import os, sys, warnings
sys.path.append('ComfyUI')
warnings.filterwarnings('ignore')

import random
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch

from main import load_extra_path_config
from nodes import init_extra_nodes, CheckpointLoaderSimple, CLIPSetLastLayer, LoraLoader, ControlNetLoader, ControlNetApply, CLIPTextEncode, EmptyLatentImage, LoadImage, VAEDecode, VAEEncode, KSampler, NODE_CLASS_MAPPINGS

load_extra_path_config("arch/extra_model_paths.yaml")
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
    MODEL: str = "RealVisXL_V4.0.safetensors"
    lora: str = "JuggerCineXL2.safetensors"
    cnet_depth: str = "diffusers_xl_depth_full.safetensors"
    cnet_canny: str = "diffusers_xl_canny_full.safetensors"
    upscale: str = "RealESRGAN_x2.pth"

    prompt: str = "a photograph of a red starfish, ((high resolution, high-resolution, cinematic, technicolor, film grain, analog, 70mm, 8K, IMAX, Nat Geo, DSLR))"
    negative: str = "worst quality, low quality, low-res, low details, cropped, blurred, defocus, bokeh, oversaturated, undersaturated, overexposed, underexposed, letterbox, aspect ratio, formatted, jpeg artefacts, draft, glitch, error, deformed, distorted, disfigured, duplicated, bad proportions"

    factor: int = 1
    w, h = 2048 // factor, 1152 // factor

    sampler: str =  "dpmpp_2m_sde" # "dpmpp_sde" "dpmpp_2m"
    scheduler: str = "karras"
    num_images: int = 1
    infer_steps: int = 20
    denoise: float = 0.8
    cfg_scale: float = 7.0

    enable_img2img: bool = False
    enable_lora: bool = False
    enable_controlnet: bool = False
    enable_upscale: bool = False

    VERSION: str = "v001"
    SOURCE: str = f"/mnt/vanguard/PLAYHOUSE/render/{VERSION}/"
    filename: str = f"output_{VERSION}"
    albedo: str = f"{SOURCE}{filename}_albedo.png"
    depth: str = f"{SOURCE}{filename}_depth.png"
    # normal: str = f"{SOURCE}{filename}_normal.png"
    curvature: str = f"{SOURCE}{filename}_curvature.png"

    depth_strength: float = 0.75
    canny_strength: float = 0.25
    lora_model: float = 0.75
    lora_clip: float = 0.75

def main():
    config = HyperConfig()

    with torch.no_grad():
        ckpt = CheckpointLoaderSimple().load_checkpoint(ckpt_name=config.MODEL)

        if config.enable_lora:
            clipsetlastlayer = CLIPSetLastLayer().set_last_layer(stop_at_clip_layer=-1, clip=ckpt[1])
            loraloader = LoraLoader().load_lora(lora_name=config.lora, strength_model=config.lora_model, strength_clip=config.lora_clip, model=ckpt[0], clip=clipsetlastlayer[0])
            clip_model = loraloader[1]
        else:
            clip_model = ckpt[1]

        image = LoadImage()
        if config.enable_img2img:
            cd = image.load_image(image=config.albedo)
        if config.enable_controlnet:
            z = image.load_image(image=config.depth)
            curv = image.load_image(image=config.curvature)

        if config.enable_img2img:
            enc = VAEEncode().encode(pixels=cd[0], vae=ckpt[2])
        else:
            enc = EmptyLatentImage().generate(width=config.w, height=config.h, batch_size=1)

        clipencode = CLIPTextEncode()
        clipencode_prompt = clipencode.encode(text=config.prompt, clip=clip_model)
        clipencode_negative = clipencode.encode(text=config.negative, clip=clip_model)

        if config.enable_controlnet:
            controlnet = ControlNetApply()
            controlnetloader = ControlNetLoader()
            controlnetloader_depth = controlnetloader.load_controlnet(control_net_name=config.cnet_depth)
            controlnetloader_canny = controlnetloader.load_controlnet(control_net_name=config.cnet_canny)

            clip_depth = controlnet.apply_controlnet(strength=config.depth_strength, conditioning=clipencode_prompt[0], control_net=controlnetloader_depth[0], image=z[0])
            clip_canny = controlnet.apply_controlnet(strength=config.canny_strength, conditioning=clip_depth[0], control_net=controlnetloader_canny[0], image=curv[0])
            emb = clip_canny[0]
        else:
            emb = clipencode_prompt[0]

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
                positive=emb,
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