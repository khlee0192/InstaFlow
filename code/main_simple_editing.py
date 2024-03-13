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
        num_inversion_steps=args.num_inversion_steps, num_inference_steps=args.num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False,
        decoder_inv_steps=30,
        forward_steps=100,
        tuning_steps=10,
        pnp_adjust=False,
        reg_coeff=0,
    )

    seed = np.random.randint(0, 2**32)

    generator = torch.manual_seed(seed)
    output, latents, original_latents = insta_pipe(
        prompt=args.edit_prompt, 
        num_inference_steps=args.num_inference_steps, 
        guidance_scale=guidance_scale, 
        generator=generator, 
        latents=recon_latents
    )

    image = output.images[0]
    image.show()

    recon_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(recon_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])

    print("..Editing task Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument("--image_number", default=3, type=int)
    parser.add_argument('--inverse_prompt', default="A high-resolution photo of an orange Porsche under sunshine", type=str)
    parser.add_argument('--edit_prompt', default="A high-resolution photo of an  Porsche under sunshine", type=str)
    parser.add_argument('--guidance_scale', default=1, type=int)
    parser.add_argument('--num_inference_steps', default=1, type=int)
    parser.add_argument('--num_inversion_steps', default=1, type=int)

    args = parser.parse_args()

    main()