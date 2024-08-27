# PLAYHOUSE [Work-In-Progress] 

**Project ID:** pLxS4aFn

<div align="center">
<picture>
  <img alt="Stage Skull" src="sample.png" width="100%">
</picture>
</div>

### Latent Diffusion
Abstract: *A **Latent Diffusion Engine (LDE)** is a machine learning model that generates images by gradually refining random noise into a coherent output. It operates in a latent space, where it reverses a diffusion process (turning noise back into data) to create high-quality results.*

## Installation
    git clone https://github.com/epochlab/PLAYHOUSE

Navigate into the `STAGE` directory and clone `ComfyUI` then install the required packages.

    git clone https://github.com/comfyanonymous/ComfyUI.git
    pip install -r requirements.txt


Run:
```
python arch/execute.py
```

## HyperConfig
Define the parameters by editing the HyperConfig class found in the `execute` script.

Parameter | Description
------- | -------
[Model](https://en.wikipedia.org/wiki/Stable_Diffusion) | The diffusion model choosen for infering images (eg. SDXL).
[LoRA](https://arxiv.org/pdf/2106.09685) | Low-Rank Adaptation; a fine-tuning technique for reducing the number of parameters trained.
[Controlnet](https://arxiv.org/pdf/2302.05543) | Controlnet; a model which controls image generation by adding extra conditions (Depth & Canny)
[Upscale](https://arxiv.org/pdf/1809.00219) | Generate high-quality, high-resolution images from lower-resolution inputs by predicting and adding finer details to the image.
[Sampler](https://arxiv.org/pdf/2211.01095) | Controls the steps and processes by which noise is gradually refined into a coherent image.
[Scheduler](https://arxiv.org/pdf/2206.00364) | Dictates the timing and magnitude of updates applied to the model, impacting how noise is reduced across iterations.
[Denoise](https://arxiv.org/pdf/2112.10752) | The amount of denoise to apply to the latent source vector.
[CFG](https://arxiv.org/pdf/2112.10752) | Classifier-free guidance scale; controls how much the image generation process follows the text prompt.

## IO

## Models
Currently `PLAYHOUSE` only supports Stable-Diffusion (1.4, 1.5, 3, SDXL).