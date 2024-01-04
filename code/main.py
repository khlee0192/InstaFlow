## This code is for genearting basic image sample, adjusted from local_gradio.py

from pipeline_rf import RectifiedFlowPipeline

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

from diffusers import StableDiffusionXLImg2ImgPipeline
import time
import copy
import numpy as np
import argparse
import matplotlib.pyplot as plt

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

insta_pipe = RectifiedFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16) 
#dW_dict = get_dW_and_merge(insta_pipe, lora_path="Lykon/dreamshaper-7", save_dW=False, alpha=1.0)     
insta_pipe.to("cuda")

@torch.no_grad()
def set_new_latent_and_generate_new_image(seed, prompt, randomize_seed, num_inference_steps=1, guidance_scale=0.0):
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
    images = insta_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator).images 
    inf_time = time.time() - t_s 

    img = copy.copy(np.array(images[0]))

    return images[0], inf_time, seed

def main():
    image, time, seed = set_new_latent_and_generate_new_image(args.seed, args.prompt, args.randomize_seed, num_inference_steps=1, guidance_scale=0.0)

    plt.imshow(image)
    plt.show()
    print(f"generation time : {time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=True, type=bool)
    parser.add_argument('--seed', default=2736710, type=int)
    parser.add_argument('--prompt', default="A dog in the field", type=str)

    args = parser.parse_args()

    main()