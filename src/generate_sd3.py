import argparse
import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from transformers import BitsAndBytesConfig, T5EncoderModel

from diffusers import StableDiffusion3Pipeline
from utils import generate_filename_from_prompt


def read_prompts(prompts_file: str) -> list:
    """
    Read prompts from a text file.

    Args:
        prompts_file (str): The prompts file.

    Returns:
        list: The list of prompts.
    """
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
    return prompts


def load_model(
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    device_map: str = "balanced",
    torch_dtype: torch.dtype = torch.float16,
):
    """
    Load the Stable Diffusion 3 model.

    Args:
        model_id (str): The model id.
        device_map (str): The device map.
        torch_dtype (torch.dtype): The torch data type.
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_3",
        quantization_config=quantization_config,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=text_encoder,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return pipe


def generate_image(
    prompts: str,
    pipe: StableDiffusion3Pipeline,
    negative_prompt: str = "",
    output_folder: str = "output",
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
) -> None:
    """
    Generate an image using the Stable Diffusion 3 model.

    Args:
        prompt (str): The prompt to generate the image.
        negative_prompt (str): The negative prompt to generate the image.
        output_folder (str): The output folder to save the image.
        num_inference_steps (int): The number of inference steps.
        guidance_scale (float): The guidance scale.
        device_map (str): The device map.
        torch_dtype (torch.dtype): The torch data type.
    """

    # create folder by current date
    date = datetime.now().strftime("%Y-%m-%d")
    # create the folder if not exists
    output_folder = f"{output_folder}/{date}"
    os.makedirs(output_folder, exist_ok=True)

    # Make sure you have `bitsandbytes` installed.
    for prompt in prompts:
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        fname = generate_filename_from_prompt(prompt)
        output_file = f"{output_folder}/{fname}.png"
        # save the image
        image.save(output_file)


def main(args):
    """Main function to generate images using the Stable Diffusion 3 model."""
    config = OmegaConf.load(args.sd3_config)

    pipe = load_model(config.model.params.model_id)

    # read prompts from txt file and put them as a list
    prompts = read_prompts(config.prompts.fpath)
    print("Prompts: ", prompts)

    generate_image(prompts=prompts, pipe=pipe, output_folder=config.model.output.images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd3_config", type=str, default="configs/sd3.yaml")
    arguments = parser.parse_args()

    main(arguments)
