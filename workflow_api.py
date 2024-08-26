import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    NODE_CLASS_MAPPINGS,
    VAEEncode,
    ControlNetApply,
    CLIPTextEncode,
    LoadImage,
    CheckpointLoaderSimple,
    CLIPSetLastLayer,
    LoraLoader,
    EmptyLatentImage,
    ControlNetLoader,
    KSampler,
    SaveImage,
    VAEDecode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_152 = checkpointloadersimple.load_checkpoint(
            ckpt_name="RealVisXL_V4.0.safetensors"
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_202 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-1,
            clip=get_value_at_index(checkpointloadersimple_152, 1),
        )

        loraloader = LoraLoader()
        loraloader_201 = loraloader.load_lora(
            lora_name="JuggerCineXL2.safetensors",
            strength_model=0.75,
            strength_clip=0.75,
            model=get_value_at_index(checkpointloadersimple_152, 0),
            clip=get_value_at_index(clipsetlastlayer_202, 0),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_154 = cliptextencode.encode(
            text="worst quality, low quality, low-res, low details, cropped, blurred, defocus, bokeh, oversaturated, undersaturated, overexposed, underexposed, letterbox, aspect ratio, formatted, jpeg artefacts, draft, glitch, error, deformed, distorted, disfigured, duplicated, bad proportions",
            clip=get_value_at_index(loraloader_201, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_155 = controlnetloader.load_controlnet(
            control_net_name="diffusers_xl_depth_full.safetensors"
        )

        loadimage = LoadImage()
        loadimage_157 = loadimage.load_image(image="stage_v001_depth.png")

        loadimage_184 = loadimage.load_image(image="stage_v001_albedo.png")

        vaeencode = VAEEncode()
        vaeencode_185 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_184, 0),
            vae=get_value_at_index(checkpointloadersimple_152, 2),
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_195 = upscalemodelloader.load_model(
            model_name="RealESRGAN_x2.pth"
        )

        loadimage_218 = loadimage.load_image(image="stage_v001_alpha.png")

        controlnetloader_235 = controlnetloader.load_controlnet(
            control_net_name="diffusers_xl_canny_full.safetensors"
        )

        loadimage_236 = loadimage.load_image(image="stage_v001_curvature.png")

        cliptextencode_250 = cliptextencode.encode(
            text="a photograph of a red starfish, ((high resolution, high-resolution, cinematic, technicolor, film grain, analog, 70mm, 8K, IMAX, Nat Geo, DSLR))",
            clip=get_value_at_index(loraloader_201, 1),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_253 = emptylatentimage.generate(
            width=1024, height=576, batch_size=1
        )

        checkpointloadersimple_265 = checkpointloadersimple.load_checkpoint(
            ckpt_name="RealVisXL_V4.0.safetensors"
        )

        clipsetlastlayer_266 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-1,
            clip=get_value_at_index(checkpointloadersimple_265, 1),
        )

        loraloader_267 = loraloader.load_lora(
            lora_name="JuggerCineXL2.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_265, 0),
            clip=get_value_at_index(clipsetlastlayer_266, 0),
        )

        controlnetapply = ControlNetApply()
        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        latentcompositemasked = NODE_CLASS_MAPPINGS["LatentCompositeMasked"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        saveimage = SaveImage()

        for q in range(10):
            controlnetapply_156 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(cliptextencode_250, 0),
                control_net=get_value_at_index(controlnetloader_155, 0),
                image=get_value_at_index(loadimage_157, 0),
            )

            controlnetapply_242 = controlnetapply.apply_controlnet(
                strength=0.5,
                conditioning=get_value_at_index(controlnetapply_156, 0),
                control_net=get_value_at_index(controlnetloader_235, 0),
                image=get_value_at_index(loadimage_236, 0),
            )

            imagetomask_219 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(loadimage_218, 0)
            )

            latentcompositemasked_262 = latentcompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(emptylatentimage_253, 0),
                source=get_value_at_index(vaeencode_185, 0),
                mask=get_value_at_index(imagetomask_219, 0),
            )

            ksampler_151 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=0.75,
                model=get_value_at_index(loraloader_201, 0),
                positive=get_value_at_index(controlnetapply_242, 0),
                negative=get_value_at_index(cliptextencode_154, 0),
                latent_image=get_value_at_index(latentcompositemasked_262, 0),
            )

            vaedecode_158 = vaedecode.decode(
                samples=get_value_at_index(ksampler_151, 0),
                vae=get_value_at_index(checkpointloadersimple_152, 2),
            )

            imageupscalewithmodel_194 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_195, 0),
                image=get_value_at_index(vaedecode_158, 0),
            )

            saveimage_248 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(imageupscalewithmodel_194, 0),
            )

            vaedecode_257 = vaedecode.decode(
                samples=get_value_at_index(latentcompositemasked_262, 0),
                vae=get_value_at_index(checkpointloadersimple_152, 2),
            )


if __name__ == "__main__":
    main()
