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

# SDXL Native Resolutions | 1024x1024 1152x896 896x1152 1216x832 832x1216 1344x768 768x1344 1536x640 640x1536
# 2.35 Aspect ratio | 1280x545, 1920x816, 2048x871

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

    prompt: str = "a photograph of a human skull, ((high resolution, high-resolution, cinematic, technicolor, film grain, analog, 70mm, 8K, IMAX, Nat Geo, DSLR))"
    negative: str = "worst quality, low quality, low-res, low details, cropped, blurred, defocus, bokeh, oversaturated, undersaturated, overexposed, underexposed, letterbox, aspect ratio, formatted, jpeg artefacts, draft, glitch, error, deformed, distorted, disfigured, duplicated, bad proportions"

    VERSION: str = "v001"
    SOURCE: str = f"/mnt/vanguard/STAGE/render/{VERSION}/"
    filename: str = f"stage_{VERSION}_"
    albedo: str = f"{SOURCE}{filename}albedo.png"
    depth: str = f"{SOURCE}{filename}depth.png"
    curvature: str = f"{SOURCE}{filename}curvature.png"

    factor: int = 2
    w, h = 2048 // factor, 1152 // factor

    depth_strength: float = 0.75
    canny_strength: float = 0.25
    lora_model: float = 0.75
    lora_clip: float = 0.75

    sampler: str = "dpmpp_2m_sde" # "dpmpp_sde"  "dpmpp_2m"
    scheduler: str = "karras"
    num_images: int = 1
    infer_steps: int = 20
    denoise: float = 0.5
    cfg_scale: float = 8.0

    enable_img2img: bool = True
    enable_lora: bool = False
    enable_controlnet: bool = True
    enable_upscale: bool = True

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
        cd = image.load_image(image=config.albedo)
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