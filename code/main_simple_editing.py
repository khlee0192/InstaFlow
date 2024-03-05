## This code is for editing real image

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

from PIL import Image

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None) 
insta_pipe.to("cuda")

def main():
    print("Start Editing task of single image")

    # Prepare Image
    image_path = "./edit_image/" + str(args.image_number) + ".png"
    img = Image.open(image_path).convert('RGB').resize((512, 512))
    img_np = np.array(img)
    img_tensor = ToTensor()(img_np).cuda().unsqueeze(0).permute(0, 2, 3, 1)

    # Set varaibels
    guidance_scale = float(args.guidance_scale)

    # Perform inversion to obtain noise
    recon_latents, recon_images = insta_pipe.exact_inversion(
        prompt=args.inverse_prompt,
        #latents=latents,
        image=img_tensor,
        input_type="dec_inv",
        num_inversion_steps=1, num_inference_steps=1, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False,
        decoder_inv_steps=30,
        forward_steps=100,
        tuning_steps=10,
        pnp_adjust=False,
        reg_coeff=1,
    )

    print("..Editing task Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument("--image_number", default=1, type=int)
    parser.add_argument('--inverse_prompt', default="", type=str)
    parser.add_argument('--edit_prompt', default="", type=str)
    parser.add_argument('--guidance_scale', default=3, type=int)

    args = parser.parse_args()

    main()