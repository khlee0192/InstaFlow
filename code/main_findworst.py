## This code is for genearting basic image sample, adjusted from local_gradio.py

from pipeline_rf import RectifiedInversableFlowPipeline

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

import wandb
import time
import copy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import math

from utils import *
from dataset import *

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None) 
insta_pipe.to("cuda")

@torch.no_grad()
def single_exp(seed, prompt, inversion_prompt, randomize_seed,
                                                  decoder_inv_steps=30, forward_steps=100, tuning_steps=10, tuning_lr=0.01, reg_coeff=0,
                                                  num_inference_steps=1, num_inversion_steps=1, guidance_scale=3.0):
    if randomize_seed:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale) 

    # In this case, we want to compare latents vs encoder latents
    generator = torch.manual_seed(seed)
    output, latents, original_latents = insta_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
    original_images = output.images
    original_image = original_images[0]
    original_array = torch.Tensor(insta_pipe.image_processor.pil_to_numpy(original_image)).permute(0, 3, 1, 2)

    original_array = original_array.to('cuda')
    encoded_latents = insta_pipe.get_image_latents(original_array, sample=False)

    test_decode_latents = insta_pipe.decode_latents(latents)
    test_decode_encoded_latents = insta_pipe.decode_latents(encoded_latents)

    error = (latents - encoded_latents).norm()**2/(latents).norm()**2

    # Scale seems to not match
    return latents, encoded_latents, error

def main():
    dataset, prompt_key = get_dataset(args.dataset)

    if args.with_tracking:
        wandb.init(project='Encoder Decoder Test', name=args.run_name)
        table = wandb.Table(columns=['image number', 'error', 'seed'])
        wandb.config.update(args)

    for i in tqdm(range(args.start, args.end)):
        seed = args.gen_seed
        
        prompt = dataset[i][prompt_key]
        # while inversion prompt is just the same,
        inversion_prompt = prompt

        latents, encoded_latents, error = single_exp(seed, prompt, inversion_prompt,
                    args.randomize_seed, decoder_inv_steps=args.decoder_inv_steps, forward_steps=args.forward_steps, tuning_steps=args.tuning_steps, tuning_lr=args.tuning_lr, reg_coeff=args.reg_coeff, num_inference_steps=1, num_inversion_steps=1, guidance_scale=args.guidance_scale)

        if args.with_tracking:
            table.add_data(i, error, seed)

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--guidance_scale', default=1, type=float)
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--decoder_inv_steps', default=50, type=int)
    parser.add_argument('--forward_steps', default=100, type=int)
    parser.add_argument('--tuning_steps', default=20, type=int)
    parser.add_argument('--tuning_lr', default=0.01, type=float)
    parser.add_argument('--reg_coeff', default=1.0, type=float)
    parser.add_argument('--with_tracking', action='store_true', help="track with wandb")

    args = parser.parse_args()

    main()