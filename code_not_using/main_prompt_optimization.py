## This code is for genearting basic image sample, adjusted from local_gradio.py

from pipeline_rf import RectifiedFlowPipeline
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

from datetime import datetime

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None)
insta_pipe.to("cuda")

@torch.no_grad()
def inverse_text_optimizer(seed, prompt, randomize_seed, num_inference_steps=1, num_inversion_steps=1, guidance_scale=3.0):
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

    inf_time = time.time() - t_s

    recon_latents, recon_images = insta_pipe.prompt_optimization(
        prompt=prompt, 
        latents=latents, 
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False
        )
    
    print(f"TOT of inversion {(recon_latents - original_latents).norm()/original_latents.norm()}")

    # Visualizing noise
    original_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(original_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    recon_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(recon_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])

    original_latents_visualized = np.squeeze(original_latents_visualized)
    recon_latents_visualized = np.squeeze(recon_latents_visualized)

    # Obtain image, calculate OTO
    original_image = original_images[0]
    recon_image = recon_images[0]
    original_array = insta_pipe.image_processor.pil_to_numpy(original_image)
    recon_array = insta_pipe.image_processor.pil_to_numpy(recon_image)
    diff = np.linalg.norm(original_array - recon_array)

    print(f"OTO of inversion {diff/np.linalg.norm(original_array)}")

    print(seed, num_inference_steps, guidance_scale)

    return original_image, recon_image, original_latents_visualized, recon_latents_visualized, inf_time, seed

def main():
    image, recon_image, latents, recon_latents, time, seed = inverse_text_optimizer(
        args.seed, args.prompt, args.randomize_seed, 
        num_inference_steps=1, num_inversion_steps=1,
        guidance_scale=1.0
    )

    # Create a figure and axes for the grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Display the first PIL image
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off') 
    axs[0, 0].set_title('Image')

    # Display the second PIL image
    axs[0, 1].imshow(recon_image)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Reconstructed Image')

    # Display the first numpy array (latents) as an RGB image
    axs[1, 0].imshow(latents)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Latents')

    # Display the second numpy array (recon_latents) as an RGB image
    axs[1, 1].imshow(recon_latents)
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Reconstructed Latents')

    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f"plot_{current_time}.png")
    #plt.show()

    print(f"generation time : {time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--seed', default=1049551870, type=int)
    parser.add_argument('--prompt', default="castle and river, High quality", type=str)
    #parser.add_argument('--prompt', default="A high resolution photo of an yellow porsche under sunshine", type=str)

    args = parser.parse_args()

    main()