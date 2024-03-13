## This code is for genearting basic image sample, adjusted from local_gradio.py

from pipeline_rf import RectifiedInversableFlowPipeline

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

from diffusers import StableDiffusionXLImg2ImgPipeline
import time
import copy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from utils import *

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None) 
insta_pipe.to("cuda")

@torch.no_grad()
def generate_image(seed, prompt, randomize_seed, num_inference_steps, guidance_scale):
    print('Generate with input seed')
    if randomize_seed:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale) 

    t_s = time.time()
    generator = torch.manual_seed(seed)
    output, latents, original_latents = insta_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
    original_images = output.images
    original_image = original_images[0]
    original_array = insta_pipe.image_processor.pil_to_numpy(original_image)

    inf_time = time.time() - t_s

    return original_image, prompt, inf_time, seed

def main():
    image, prompt, inf_time, seed = generate_image(
        seed=args.seed, 
        prompt=args.prompt, 
        randomize_seed=args.randomize_seed, 
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    ) 

    file_name = prompt + " " + str(seed) + ".png"
    image.save(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=True, type=bool)
    parser.add_argument('--seed', default=2993109412, type=int)
    parser.add_argument('--prompt', default="A high-resolution photo of an blue Porsche under sunshine", type=str)
    parser.add_argument('--num_inference_steps', default=10, type=int)
    parser.add_argument('--guidance_scale', default=1, type=float)

    args = parser.parse_args()

    main()