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

insta_pipe = RectifiedInversableFlowPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16, safety_checker=None) 
insta_pipe.to("cuda")

@torch.no_grad()
def single_exp(seed, prompt, inversion_prompt, randomize_seed,
                                                  decoder_inv_steps=30, decoder_lr=0.1, forward_steps=100, tuning_steps=10, tuning_lr=0.01, reg_coeff=0,
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

    if args.test_beta:
        recon_images, recon_zo, recon_latents, output_loss, peak_gpu_allocated, dec_inv_time, extra_outputs, extra_outputs_another, another_output = insta_pipe.exact_inversion(
        prompt=inversion_prompt,
        latents=latents,
        image=original_array,
        input_type=args.inversion_type,
        decoder_use_float=args.use_float,
        test_beta=args.test_beta,
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=False,
        use_random_initial_noise=False,
        decoder_inv_steps=decoder_inv_steps,
        decoder_lr=decoder_lr,
        decoder_adam=args.decoder_adam,
        forward_steps=forward_steps,
        tuning_steps=0,
        tuning_lr=tuning_lr,
        pnp_adjust=False,
        reg_coeff=reg_coeff,
    )
    
    else:
        recon_images, recon_zo, recon_latents, output_loss, peak_gpu_allocated, dec_inv_time = insta_pipe.exact_inversion(
        prompt=inversion_prompt,
        latents=latents,
        image=original_array,
        input_type=args.inversion_type,
        decoder_use_float=args.use_float,
        test_beta=args.test_beta,
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=False,
        use_random_initial_noise=False,
        decoder_inv_steps=decoder_inv_steps,
        decoder_lr=decoder_lr,
        forward_steps=forward_steps,
        tuning_steps=0,
        tuning_lr=tuning_lr,
        pnp_adjust=False,
        reg_coeff=reg_coeff,
    )

    # inv_time = time.time() - t_inv
    inv_time = dec_inv_time

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
    error_middle = ((recon_zo-latents).norm().item()**2)/((latents).norm().item()**2)

    # max GPU
    peak_gpu_allocated = peak_gpu_allocated / (1024**3)

    if args.test_beta:
        return original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, peak_gpu_allocated, inf_time, inv_time, seed, extra_outputs, extra_outputs_another, another_output
    else:
        return original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, peak_gpu_allocated, inf_time, inv_time, seed

def main():
    dataset, prompt_key = get_dataset(args.dataset)

    if args.with_tracking:
        wandb.init(project='240512 work on KM momentum', name=args.run_name)
        wandb.config.update(args)
        table = wandb.Table(columns=['original_image', 'recon_image', 'diff_image', 'original_latents_visualized', 'recon_latents_visualized', 'diff_latents_visualized', 'error_TOT', 'error_OTO', 'error_middle', 'output_loss', 'dec_inv memory', 'inf_time', 'inv_time', 'seed', 'prompt', 'cocercivity_process', 'cocercivity_inf'])

    TOT_list = []
    OTO_list = []
    middle_list = []
    invtime_list = []
    loss_list = []
    memory_list = []

    if args.test_beta:
        cocercivity_rate_results = torch.zeros((args.end - args.start), args.decoder_inv_steps)
        cocercivity_rate_results_another = torch.zeros((args.end - args.start), args.decoder_inv_steps)
        z_list_results = torch.zeros((args.end - args.start), args.decoder_inv_steps)

    mode = 2

    if mode == 0:
        i = 4
        j = 5

        row_list = ["A realistic image of ", "An oil painting of ", "A digital illustration of ", "A watercolor art of "]
        col_list = ["a panda eating bamboo on a rock.", "a lion roaring in the grassland.", "an eagle soaring above the mountains.", "a dolphin swimming in the crystal-clear ocean.", "a koala munching eucalyptus leaves in a tree."]

        test_list = [row + col for row in row_list for col in col_list]

        for i in range(len(test_list)):
            seed = args.gen_seed
            
            prompt = test_list[i]
            # while inversion prompt is just the same,
            inversion_prompt = prompt

            original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed = single_exp(seed, prompt, inversion_prompt,
                        args.randomize_seed, decoder_inv_steps=args.decoder_inv_steps, forward_steps=args.forward_steps, tuning_steps=args.tuning_steps, tuning_lr=args.tuning_lr, reg_coeff=args.reg_coeff, num_inference_steps=1, num_inversion_steps=1, guidance_scale=args.guidance_scale)
            
            # TOT_list.append(10*math.log10(error_TOT))
            # OTO_list.append(10*math.log10(error_OTO))
            # middle_list.append(10*math.log10(error_middle))
            # invtime_list.append(inv_time)
            # loss_list.append(10*math.log10(output_loss.cpu()))

            TOT_list.append(error_TOT)
            OTO_list.append(error_OTO)
            middle_list.append(error_middle)
            invtime_list.append(inv_time)
            loss_list.append(output_loss.cpu())

            if args.with_tracking:
                table.add_data(wandb.Image(original_image), wandb.Image(recon_image), wandb.Image(diff_image), wandb.Image(original_latents_visualized), wandb.Image(recon_latents_visualized), wandb.Image(diff_latents_visualized), error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed, prompt)        
    elif mode == 1:
        worst_list = [890, 69, 647, 237, 160, 521, 658, 820, 837, 900, 214, 759, 120, 476, 879, 165, 13, 904, 859, 671, 586, 275, 988, 322, 221, 381, 995, 72, 28, 201, 433, 829, 122, 55, 477, 345, 584, 871, 550, 397, 383, 798, 915, 803, 408, 965, 739, 181, 347, 472, 978, 790, 998, 432, 729, 451, 240, 126, 501, 851, 811, 170, 371, 231, 705, 741, 94, 835, 414, 694, 832, 121, 777, 372, 616, 300, 877, 61, 236, 774, 688, 460, 843, 806, 270, 666, 592, 502, 363, 548, 864, 404, 508, 860, 700, 224, 781, 769, 426, 986, 522, 23, 470, 987, 899, 881, 740, 656, 186, 256, 655, 912, 861, 848, 154, 967, 691, 117, 37, 264, 198, 913, 730, 317, 156, 957, 687, 872, 365, 108, 516, 461, 510, 801, 618, 778, 643, 854, 306, 421, 152, 503, 92, 479, 393, 888, 511, 484, 47, 842, 436, 17, 355, 680, 682, 980, 233, 614, 731, 85, 629, 738, 413, 631, 540, 99, 29, 815, 767, 336, 541, 307, 576, 743, 624, 654, 515, 929, 81, 289, 368, 690, 78, 204, 312, 882, 329, 44, 651, 692, 97, 603, 945, 765, 539, 239, 876, 914, 825, 606, 841, 640, 757, 836, 590, 40, 532, 710, 35, 203, 259, 956, 202, 21, 660, 269, 437, 314, 313, 139, 579, 679, 610, 812, 155, 57, 196, 113, 302, 926, 911, 672, 278, 80, 378, 136, 669, 942, 758, 752, 964, 41, 804, 716, 84, 794, 189, 564, 116, 661, 481, 290, 681, 684, 457, 595, 391, 178, 34, 960, 205, 444, 824, 443, 663, 678, 784, 167, 191, 906, 71, 981, 284, 568, 737, 632, 575, 838, 107, 305, 962, 380, 599, 760, 598, 905, 328, 497, 392, 907, 707, 814, 272, 377, 823, 250, 143, 109, 0, 133, 887, 62, 746, 667, 762, 469, 218, 454, 517, 20, 821, 52, 852, 258, 880, 424, 936, 220, 609, 526, 500, 369, 870, 135, 213, 265, 818, 157, 602, 916, 619, 462, 976, 467, 560, 933, 268, 567, 797, 412, 427, 725, 519, 722, 188, 45, 182, 172, 104, 185, 974, 356, 553, 361, 785, 351, 387, 206, 535, 173, 753, 318, 474, 868, 908, 245, 25, 354, 959, 288, 291, 296, 693, 384, 928, 282, 809, 247, 898, 96, 415, 492, 969, 891, 137, 95, 64, 505, 924, 169, 630, 670, 105, 947, 985, 352, 628, 229, 834, 127, 5, 16, 309, 782, 73, 161, 36, 990, 162, 298, 786, 48, 892, 111, 183, 331, 554, 587, 297, 4, 605, 70, 455, 875, 755, 210, 440, 428, 346, 68, 799, 403, 399, 339, 717, 650, 38, 559, 822, 831, 673, 713, 253, 252, 920, 593, 704, 465, 187, 367, 341, 374, 719, 874, 766, 558, 507, 531, 358, 480, 416, 483, 706, 645, 238, 952, 749, 422, 512, 395, 992, 39, 582, 613, 542, 733, 280, 919, 787, 863, 634, 491, 973, 764, 260, 463, 901, 138, 580, 131, 583, 148, 897, 633, 102, 756, 266, 330, 376, 728, 292, 277, 215, 9, 217, 325, 917, 411, 249, 763, 847, 498, 488, 850, 359, 937, 398, 19, 26, 637, 350, 190, 208, 552, 984, 726, 925, 342, 466, 150, 573, 635, 60, 709, 458, 417, 971, 353, 979, 611, 596, 982, 473, 536, 447, 776, 944, 703, 285, 197, 435, 495, 459, 396, 622, 668, 581, 159, 525, 578, 228, 572, 853, 808, 555, 106, 394, 711, 129, 222, 723, 22, 551, 293, 63, 577, 83, 419, 370, 93, 283, 286, 287, 32, 337, 538, 281, 423, 273, 499, 334, 115, 954, 561, 295, 695, 255, 718, 884, 748, 727, 464, 563, 12, 793, 98, 966, 389, 14, 486, 274, 796, 471, 970, 58, 246, 487, 780, 644, 90, 3, 885, 963, 849, 817, 216, 219, 158, 659, 518, 930, 223, 386, 639, 53, 6, 446, 626, 11, 537, 715, 335, 569, 701, 674, 320, 171, 452, 184, 180, 248, 207, 840, 975, 168, 123, 82, 657, 953, 163, 56, 410, 734, 67, 103, 570, 429, 385, 623, 747, 263, 999, 89, 349, 770, 931, 49, 638, 390, 409, 373, 805, 873, 886, 621, 406, 615, 59, 324, 789, 816, 783, 449, 130, 54, 938, 607, 529, 303, 844, 597, 601, 652, 112, 489, 230, 468, 958, 686, 996, 114, 276, 955, 164, 636, 425, 571, 941, 299, 735, 528, 434, 883, 134, 124, 845, 662, 177, 195, 617, 745, 810, 894, 321, 994, 683, 677, 243, 125, 257, 724, 333, 544, 142, 308, 375, 833, 736, 997, 918, 697, 927, 826, 174, 1, 15, 608, 232, 665, 110, 262, 513, 194, 589, 961, 241, 895, 574, 921, 788, 732, 388, 442, 744, 151, 430, 983, 175, 301, 856, 557, 46, 65, 119, 405, 649, 972, 153, 676, 946, 866, 588, 179, 79, 24, 664, 791, 839, 348, 546, 566, 496, 950, 620, 549, 902, 364, 2, 846, 7, 708, 77, 865, 523, 379, 234, 475, 903, 696, 146, 896, 485, 935, 360, 323, 400, 685, 200, 545, 235, 141, 132, 326, 802, 939, 86, 562, 585, 923, 527, 968, 612, 698, 910, 742, 989, 87, 604, 828, 226, 456, 772, 251, 556, 977, 779, 675, 534, 494, 565, 813, 315, 641, 227, 750, 720, 340, 294, 594, 858, 795, 514, 212, 893, 431, 382, 176, 648, 279, 775, 18, 338, 100, 211, 827, 42, 800, 689, 699, 909, 439, 948, 401, 714, 244, 771, 140, 934, 932, 144, 943, 922, 889, 993, 591, 482, 830, 51, 311, 118, 10, 91, 145, 33, 453, 751, 721, 441, 101, 867, 445, 807, 88, 319, 490, 75, 344, 855, 857, 653, 438, 366, 402, 949, 543, 43, 193, 27, 261, 712, 31, 547, 991, 327, 773, 128, 768, 524, 509, 530, 506, 316, 869, 493, 149, 625, 627, 878, 271, 192, 420, 147, 8, 862, 332, 533, 761, 418, 225, 74, 199, 951, 30, 254, 209, 642, 407, 362, 304, 76, 792, 267, 504, 50, 166, 646, 66, 940, 702, 310, 819, 478, 357, 754, 600, 242, 343, 448, 520, 450, ]
        worst_list = worst_list[args.start:args.end]

        for i in tqdm(worst_list):
            seed = args.gen_seed
            
            prompt = dataset[i][prompt_key]
            # while inversion prompt is just the same,
            inversion_prompt = prompt

            original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed = single_exp(seed, prompt, inversion_prompt,
                        args.randomize_seed, decoder_inv_steps=args.decoder_inv_steps, forward_steps=args.forward_steps, tuning_steps=args.tuning_steps, tuning_lr=args.tuning_lr, reg_coeff=args.reg_coeff, num_inference_steps=1, num_inversion_steps=1, guidance_scale=args.guidance_scale)
            
            # TOT_list.append(10*math.log10(error_TOT))
            # OTO_list.append(10*math.log10(error_OTO))
            # middle_list.append(10*math.log10(error_middle))
            # invtime_list.append(inv_time)
            # loss_list.append(10*math.log10(output_loss.cpu()))

            TOT_list.append(error_TOT)
            OTO_list.append(error_OTO)
            middle_list.append(error_middle)
            invtime_list.append(inv_time)
            loss_list.append(output_loss.cpu())

            if args.with_tracking:
                table.add_data(wandb.Image(original_image), wandb.Image(recon_image), wandb.Image(diff_image), wandb.Image(original_latents_visualized), wandb.Image(recon_latents_visualized), wandb.Image(diff_latents_visualized), error_TOT, error_OTO, error_middle, output_loss, inf_time, inv_time, seed, prompt)
    else:
        for i in tqdm(range(args.start, args.end)):
            seed = args.gen_seed
            
            prompt = dataset[i][prompt_key] 
            inversion_prompt = prompt

            original_image, recon_image, diff_image, original_latents_visualized, recon_latents_visualized, diff_latents_visualized, error_TOT, error_OTO, error_middle, output_loss, peak_gpu_allocated, inf_time, inv_time, seed, extra_outputs, extra_outputs_another, another_output = single_exp(seed, prompt, inversion_prompt,
                        args.randomize_seed, decoder_inv_steps=args.decoder_inv_steps, decoder_lr=args.decoder_lr, forward_steps=args.forward_steps, tuning_steps=args.tuning_steps, tuning_lr=args.tuning_lr, reg_coeff=args.reg_coeff, num_inference_steps=1, num_inversion_steps=1, guidance_scale=args.guidance_scale)
            
            # TOT_list.append(10*math.log10(error_TOT))
            # OTO_list.append(10*math.log10(error_OTO))
            # middle_list.append(10*math.log10(error_middle))
            # invtime_list.append(inv_time)
            # loss_list.append(10*math.log10(output_loss.cpu()))

            TOT_list.append(error_TOT)
            OTO_list.append(error_OTO)
            middle_list.append(error_middle)
            invtime_list.append(inv_time)
            if output_loss == 0:
                loss_list.append(0)
            else:
                loss_list.append(output_loss.cpu())
            memory_list.append(peak_gpu_allocated)

            cocercivity_process = torch.min(extra_outputs).item()
            cocercivity_inf = torch.min(extra_outputs_another).item()

            if args.with_tracking:
                table.add_data(wandb.Image(original_image), wandb.Image(recon_image), wandb.Image(diff_image), wandb.Image(original_latents_visualized), wandb.Image(recon_latents_visualized), wandb.Image(diff_latents_visualized), error_TOT, error_OTO, error_middle, output_loss, peak_gpu_allocated, inf_time, inv_time, seed, prompt, cocercivity_process, cocercivity_inf)

            if args.test_beta:
                cocercivity_rate_results[i] = extra_outputs
                cocercivity_rate_results_another[i] = extra_outputs_another
                z_list_results[i] = another_output

    if args.test_beta:
        title_first = "vanila_process_" + time.strftime("%Y%m%d-%H%M%S") + ".pt"
        title_second = "vanila_inf_" + time.strftime("%Y%m%d-%H%M%S") + ".pt"
        title_third = "z_list_" + time.strftime("%Y%m%d-%H%M%S") + ".pt"
        torch.save(cocercivity_rate_results, title_first)
        torch.save(cocercivity_rate_results_another, title_second)
        torch.save(z_list_results, title_third)
    
    
    mean_TOT = 10*math.log10(np.mean(np.array(TOT_list).flatten()))
    std_TOT = np.std(np.array(TOT_list).flatten())
    mean_OTO = 10*math.log10(np.mean(np.array(OTO_list).flatten()))
    std_OTO = np.std(np.array(OTO_list).flatten())
    mean_middle_error = 10*math.log10(np.mean(np.array(middle_list).flatten()))
    std_middle_error = np.std(np.array(middle_list).flatten())
    inv_time_avg = np.mean(np.array(invtime_list).flatten())
    mean_loss = np.mean(np.array(loss_list).flatten())
    std_loss = np.std(np.array(loss_list).flatten())
    mean_memory_usage = np.mean(np.array(memory_list))

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'mean_T0T' : mean_TOT,'std_T0T' : std_TOT,'mean_0T0' : mean_OTO,'std_0T0' : std_OTO, 'mean_middle' : mean_middle_error, 'std_middle' : std_middle_error, 'mean_loss' : mean_loss, 'std_loss' : std_loss, 'guidance_scale': args.guidance_scale, 'inv_time': inv_time_avg, 'prompt': prompt, 'mean_memory_usage': mean_memory_usage, 'decoder lr' : args.decoder_lr})
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
    parser.add_argument('--inversion_type', default="dec_inv", type=str)
    parser.add_argument('--use_float', action='store_true')
    parser.add_argument('--test_beta', action='store_true')
    parser.add_argument('--decoder_inv_steps', default=100, type=int)
    parser.add_argument('--decoder_lr', default=0.1, type=float)
    parser.add_argument('--decoder_adam', default=True, type=bool)
    parser.add_argument('--forward_steps', default=1, type=int)
    parser.add_argument('--tuning_steps', default=0, type=int)
    parser.add_argument('--tuning_lr', default=0.0, type=float)
    parser.add_argument('--reg_coeff', default=0.0, type=float)
    parser.add_argument('--with_tracking', action='store_true', help="track with wandb")

    args = parser.parse_args()

    main()