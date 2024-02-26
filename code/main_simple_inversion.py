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
def set_and_generate_image_then_reverse(seed, prompt, inversion_prompt, randomize_seed, num_inference_steps=1, num_inversion_steps=1, guidance_scale=3.0, plot_dist=False):
    print('Generate with input seed')
    negative_prompt=""
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

    recon_latents, recon_images = insta_pipe.exact_inversion(
        prompt=inversion_prompt,
        latents=latents,
        image=original_array,
        input_type="answer",
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False,
        decoder_inv_steps=10,
        forward_steps=10,
        tuning_steps=0,
        pnp_adjust=False,
        reg_coeff=1,
        )
    
    recon_latents_np = recon_latents.cpu().detach().numpy()
    original_latents_np = original_latents.cpu().detach().numpy()
    print(f"TOT of inversion { (np.linalg.norm(recon_latents_np-original_latents_np)**2)/np.linalg.norm(original_latents_np)**2}")

    # Visualizing noise and difference
    original_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(original_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    recon_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(recon_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])

    diff_latents = original_latents - recon_latents
    diff_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(diff_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])

    original_latents_visualized = np.squeeze(original_latents_visualized)
    recon_latents_visualized = np.squeeze(recon_latents_visualized)
    diff_latents_visualized = np.squeeze(diff_latents_visualized)

    # Obtain image, calculate OTO
    recon_image = recon_images[0]
    recon_array = insta_pipe.image_processor.pil_to_numpy(recon_image)
    
    # calculate difference of images
    diff = original_array - recon_array
    diff_image = insta_pipe.image_processor.postprocess(torch.Tensor(diff).permute(0, 3, 1, 2), output_type="pil")

    print(f"OTO of inversion {(np.linalg.norm(original_array - recon_array)**2)/np.linalg.norm(original_array)**2}")

    print(seed, num_inference_steps, guidance_scale)

    original_latents_data = original_latents.cpu()[0].flatten()
    recon_latents_data = recon_latents.cpu()[0].flatten()

    # Section : Test with value
    print(f"original latents (mean/std) : {original_latents.mean().item():.5f}, {original_latents.std().item():.5f}")
    print(f"reconstructed latents (mean/std) : {recon_latents.mean().item():.5f}, {recon_latents.std().item():.5f}")

    # Section : Calculating correlation of noise and latents
    # We will compare correlation of original_latents & latents, then recon_latents & latents
    latents_cpu = latents.cpu()[0]
    original_noise_cpu = original_latents.cpu()[0]
    recon_noise_cpu = recon_latents.cpu()[0]

    # Section : statistical test
    # # Shapiro-Wilk test for original_latents
    # stat_original, p_value_original = shapiro(original_latents_data)
    # print(f'Shapiro-Wilk test for Original Latents: W={stat_original:.8f}, p-value={p_value_original:.8f}')

    # # Shapiro-Wilk test for recon_latents
    # stat_recon, p_value_recon = shapiro(recon_latents_data)
    # print(f'Shapiro-Wilk test for Reconstructed Latents: W={stat_recon:.8f}, p-value={p_value_recon:.8f}')

    # Section : Check with plot distribution
    if plot_dist:
        plot_distribution(original_noise_cpu, recon_noise_cpu, latents_cpu, version="cosinesim")

    # Return values with normalization
    latents_plot = latents_cpu[0:3]
    latents_plot = (latents_plot - latents_plot.min())/(latents_plot.max() - latents_plot.min())
    original_noise_cpu = original_noise_cpu[0:3]
    original_noise_cpu = (original_noise_cpu - original_noise_cpu.min())/(original_noise_cpu.max() - original_noise_cpu.min())
    recon_noise_cpu = recon_noise_cpu[0:3]
    recon_noise_cpu = (recon_noise_cpu - recon_noise_cpu.min())/(recon_noise_cpu.max() - recon_noise_cpu.min())

    return original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, inf_time, seed

def main():
    image, recon_image, diff_image, latents, recon_latents, diff_latents, time, seed = set_and_generate_image_then_reverse(
        args.seed, args.prompt, args.inversion_prompt, args.randomize_seed, 
        num_inference_steps=1, num_inversion_steps=1,
        guidance_scale=1.0,
        plot_dist=True,
    )

    plot_and_save_image_with_difference(image, recon_image, diff_image, latents, recon_latents, diff_latents, show=False)

    print(f"generation time : {time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--seed', default=2993109412, type=int)
    parser.add_argument('--prompt', default="Generate an image of a calm lakeside scene at sunset. Showcase vibrant wildflowers in the foreground, and distant mountains painted in the soft hues of twilight. Capture the serene beauty of nature in this tranquil", type=str)
    parser.add_argument('--inversion_prompt', default="Create a serene lakeside sunset scene with vivid wildflowers in the foreground and distant mountains adorned in the gentle twilight palette. Bring to life the tranquility and beauty of nature in this peaceful image.", type=str)

    args = parser.parse_args()

    main()