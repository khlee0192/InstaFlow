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

def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    return pipe

def get_dW_and_merge(pipe_rf, lora_path='Lykon/dreamshaper-7', save_dW = False, base_sd='runwayml/stable-diffusion-v1-5', alpha=1.0):    
    # get weights of base sd models
    from diffusers import DiffusionPipeline
    _pipe = DiffusionPipeline.from_pretrained(
        base_sd, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    sd_state_dict = _pipe.unet.state_dict()
    
    # get weights of the customized sd models, e.g., the aniverse downloaded from civitai.com    
    _pipe = DiffusionPipeline.from_pretrained(
        lora_path, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    lora_unet_checkpoint = _pipe.unet.state_dict()
    
    # get the dW
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]
    
    # return and save dW dict
    if save_dW:
        save_name = lora_path.split('/')[-1] + '_dW.pt'
        torch.save(dW_dict, save_name)
        
    pipe_rf = merge_dW_to_unet(pipe_rf, dW_dict=dW_dict, alpha=alpha)
    pipe_rf.vae = _pipe.vae
    pipe_rf.text_encoder = _pipe.text_encoder
    
    return dW_dict

# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# pipe = pipe.to("cuda")

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None) 
#dW_dict = get_dW_and_merge(insta_pipe, lora_path="Lykon/dreamshaper-7", save_dW=False, alpha=1.0)     
insta_pipe.to("cuda")

@torch.no_grad()
def set_and_generate_image_then_reverse(seed, prompt, randomize_seed, num_inference_steps=1, num_inversion_steps=1, guidance_scale=0.0):
    print('Generate with input seed')
    negative_prompt=""
    if randomize_seed:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    print(seed, num_inference_steps, guidance_scale)

    t_s = time.time()
    generator = torch.manual_seed(seed)
    output, latents, original_latents = insta_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0, generator=generator)
    original_images = output.images

    inf_time = time.time() - t_s 

    recon_latents = insta_pipe.exact_inversion(prompt=prompt, latents=latents, num_inversion_steps=num_inversion_steps, guidance_scale=0.0)

    recon_output, _, _ = insta_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0, generator=generator, latents=recon_latents)
    recon_images = recon_output.images

    print(f"validation, latents mean : {original_latents.mean()}, latents std : {original_latents.std()}")
    print(f"validation, reconstructed latents mean : {recon_latents.mean()}, latents std : {recon_latents.std()}")

    print(prompt)

    print(f"TOT of inversion {(recon_latents - original_latents).norm()/original_latents.norm()}")

    # Visualizing noise
    original_latents_visualized = insta_pipe.decode_latents(original_latents)
    recon_latents_visualized = insta_pipe.decode_latents(recon_latents)
    original_latents_visualized = np.squeeze(original_latents_visualized)
    recon_latents_visualized = np.squeeze(recon_latents_visualized)

    return original_images[0], recon_images[0], original_latents_visualized, recon_latents_visualized, inf_time, seed

def main():
    image, recon_image, latents, recon_latents, time, seed = set_and_generate_image_then_reverse(
        args.seed, args.prompt, args.randomize_seed, 
        num_inference_steps=1, num_inversion_steps=1,
        guidance_scale=0.0
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
    plt.show()

    print(f"generation time : {time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--seed', default=2736710, type=int)
    parser.add_argument('--prompt', default="A dog in the field, colorful", type=str)

    args = parser.parse_args()

    main()