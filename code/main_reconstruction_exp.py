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
                                                  decoder_inv_steps=30, forward_steps=100, tuning_steps=10, reg_coeff=0,
                                                  num_inference_steps=1, num_inversion_steps=1, guidance_scale=3.0):
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

    t_inv = time.time()

    recon_images, recon_zo, recon_latents, output_loss = insta_pipe.exact_inversion(
        prompt=inversion_prompt,
        latents=latents,
        image=original_array,
        input_type="dec_inv",
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False,
        decoder_inv_steps=decoder_inv_steps,
        forward_steps=forward_steps,
        tuning_steps=tuning_steps,
        pnp_adjust=False,
        reg_coeff=reg_coeff,
    )
    
    inv_time = time.time() - t_inv

    # Visualizing noise and difference
    original_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(original_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    recon_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(recon_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    diff_latents = original_latents - recon_latents
    diff_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(diff_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    original_latents_visualized = Image.fromarray(np.squeeze(original_latents_visualized))
    recon_latents_visualized = Image.fromarray(np.squeeze(recon_latents_visualized))
    diff_latents_visualized = Image.fromarray(np.squeeze(diff_latents_visualized))

    # Obtain image, calculate OTO
    recon_image = recon_images[0]
    recon_array = insta_pipe.image_processor.pil_to_numpy(recon_image)
    
    # Calculate NMSEs
    diff = original_array - recon_array
    diff_image = insta_pipe.image_processor.postprocess(torch.Tensor(diff).permute(0, 3, 1, 2), output_type="pil")[0]

    error_TOT = (((recon_latents-original_latents).norm()/(original_latents).norm()).item())**2
    error_OTO = (np.linalg.norm(original_array - recon_array)**2)/np.linalg.norm(original_array)**2
    error_middle = (((recon_zo-latents).norm()/(latents).norm()).item())**2

    return original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed

def main():
    dataset, prompt_key = get_dataset(args.dataset)

    if args.with_tracking:
        wandb.init(project='fast_exp_with_10_error_fixed_hopefully', name=args.run_name)
        wandb.config.update(args)
        table = wandb.Table(columns=['original_image', 'recon_image', 'diff_image', 'original_latents_visualized', 'recon_latents_visualized', 'diff_latents_visualized', 'error_TOT', 'error_OTO', 'error_middle', 'output_loss', 'inf_time', 'inv_time', 'seed'])

    TOT_list = []
    OTO_list = []
    middle_list = []
    invtime_list = []
    loss_list = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        prompt = dataset[i][prompt_key]
        # while inversion prompt is just the same,
        inversion_prompt = prompt

        original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed = single_exp(seed, prompt, inversion_prompt,
                    args.randomize_seed, decoder_inv_steps=args.decoder_inv_steps, forward_steps=args.forward_steps, tuning_steps=args.tuning_steps, reg_coeff=args.reg_coeff, num_inference_steps=1, num_inversion_steps=1, guidance_scale=args.guidance_scale)
        
        TOT_list.append(10*math.log10(error_TOT))
        OTO_list.append(10*math.log10(error_OTO))
        middle_list.append(10*math.log10(error_middle))
        invtime_list.append(inv_time)
        loss_list.append(10*math.log10(output_loss.cpu()))

        if args.with_tracking:
            table.add_data(wandb.Image(original_image), wandb.Image(recon_image), wandb.Image(diff_image), wandb.Image(original_latents_visualized), wandb.Image(recon_latents_visualized), wandb.Image(diff_latents_visualized), error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed)

    mean_TOT = np.mean(np.array(TOT_list).flatten())
    std_TOT = np.std(np.array(TOT_list).flatten())
    mean_OTO = np.mean(np.array(OTO_list).flatten())
    std_OTO = np.std(np.array(OTO_list).flatten())
    mean_middle_error = np.mean(np.array(middle_list).flatten())
    std_middle_error = np.std(np.array(middle_list).flatten())
    inv_time_avg = np.mean(np.array(invtime_list).flatten())
    mean_loss = np.mean(np.array(loss_list).flatten())
    std_loss = np.std(np.array(loss_list).flatten())

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'mean_T0T' : mean_TOT,'std_T0T' : std_TOT,'mean_0T0' : mean_OTO,'std_0T0' : std_OTO, 'mean_middle' : mean_middle_error, 'std_middle' : std_middle_error, 'mean_loss' : mean_loss, 'std_loss' : std_loss, 'guidance_scale': args.guidance_scale, 'inv_time': inv_time_avg})
        wandb.finish()
    else:
        print(f"mean_TOT : {mean_TOT}, std_TOT : {std_TOT}")
        print(f"mean_OTO : {mean_OTO}, std_OTO : {std_OTO}")
        print(f"mean_middle : {mean_middle_error}, std_middle : {std_middle_error}")
        print(f"mean_loss : {mean_loss}, std_loss : {std_loss}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--run_name', default='100-100-0-0.0')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--guidance_scale', default=1, type=float)
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--decoder_inv_steps', default=50, type=int)
    parser.add_argument('--forward_steps', default=100, type=int)
    parser.add_argument('--tuning_steps', default=20, type=int)
    parser.add_argument('--reg_coeff', default=1.0, type=float)
    parser.add_argument('--with_tracking', action='store_true', help="track with wandb")

    args = parser.parse_args()

    main()