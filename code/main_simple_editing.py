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

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float32, safety_checker=None) 
insta_pipe.to("cuda")

def main():
    print("Start Editing task of single image")

    image_path = ".,/edit_image/" + str(args.image_number) + ".png"
    

    print("..Editing task Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument("--image_number", default=1, type=int)
    parser.add_argument('--inverse_prompt', default="", type=str)
    parser.add_argument('--edit_prompt', default="", type=str)

    args = parser.parse_args()

    main()