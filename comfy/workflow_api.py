#!/usr/bin/env python3

import random
from dataclasses import dataclass
import torch

from main import load_extra_path_config
from nodes import init_custom_nodes, CheckpointLoaderSimple, ControlNetLoader, ControlNetApply, CLIPTextEncode, SaveImage, LoadImage, VAEDecode, VAEEncode, KSampler, NODE_CLASS_MAPPINGS

load_extra_path_config("extra_model_paths.yaml")
init_custom_nodes()

@dataclass
class HyperConfig:
    MODEL = "RealVisXL_V4.0.safetensors"
    cnet_depth = "diffusers_xl_depth_full.safetensors"
    cnet_canny = "diffusers_xl_canny_full.safetensors"
    upscale = "RealESRGAN_x2.pth"

    prompt = "a photograph of a red starfish, ((high resolution, high-resolution, cinematic, technicolor, film grain, analog, 70mm, 8K, IMAX, Nat Geo, DSLR))"
    negative = "worst quality, low quality, low-res, low details, cropped, blurred, defocus, bokeh, oversaturated, undersaturated, overexposed, underexposed, letterbox, aspect ratio, formatted, jpeg artefacts, draft, glitch, error, deformed, distorted, disfigured, duplicated, bad proportions"

    VERSION = "v001"
    SOURCE = f"/mnt/vanguard/STAGE/render/{VERSION}/stage_{VERSION}_"
    albedo = str(SOURCE + "albedo.png")
    depth = str(SOURCE + "depth.png")
    curvature = str(SOURCE + "curvature.png")

    depth_stength = 1
    canny_stength = 0.5

    sampler = "dpmpp_2m"
    scheduler = "karras"
    num_images = 3
    infer_steps = 20
    denoise = 0.75
    cfg_scale = 8.0

def main():
    config = HyperConfig()

    with torch.inference_mode():
        ckpt = CheckpointLoaderSimple().load_checkpoint(ckpt_name=config.MODEL)
        upscaler = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]().load_model(model_name=config.upscale)

        image = LoadImage()
        cd = image.load_image(image=config.albedo)
        z = image.load_image(image=config.depth)
        curv = image.load_image(image=config.curvature)

        enc = VAEEncode().encode(pixels=cd[0], vae=ckpt[2])

        clipencode = CLIPTextEncode()
        clipencode_prompt = clipencode.encode(text=config.prompt, clip=ckpt[1])
        clipencode_negative = clipencode.encode(text=config.negative, clip=ckpt[1])

        controlnet = ControlNetApply()
        controlnetloader = ControlNetLoader()
        controlnetloader_depth = controlnetloader.load_controlnet(control_net_name=config.cnet_depth)
        controlnetloader_canny = controlnetloader.load_controlnet(control_net_name=config.cnet_canny)

        clip_depth = controlnet.apply_controlnet(strength=config.depth_stength, conditioning=clipencode_prompt[0], control_net=controlnetloader_depth[0], image=z[0])
        clip_canny = controlnet.apply_controlnet(strength=config.canny_stength, conditioning=clip_depth[0], control_net=controlnetloader_canny[0], image=curv[0])

        for _ in range(config.num_images):
            latents = KSampler().sample(
                seed=random.randint(1, 2**64),
                steps=config.infer_steps,
                cfg=config.cfg_scale,
                sampler_name=config.sampler,
                scheduler=config.scheduler,
                denoise=config.denoise,
                model=ckpt[0],
                positive=clip_canny[0],
                negative=clipencode_negative[0],
                latent_image=enc[0],
            )

            dec = VAEDecode().decode(samples=latents[0], vae=ckpt[2])
            xres = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]().upscale(upscale_model=upscaler[0], image=dec[0])
            SaveImage().save_images(filename_prefix="STAGE_diffusion", images=xres[0])

if __name__ == "__main__":
    main()