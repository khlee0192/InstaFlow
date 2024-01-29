## This code is for genearting basic image sample, adjusted from local_gradio.py

from pipeline_rf_adjusting import RectifiedInversableFlowPipeline

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

from diffusers import StableDiffusionXLImg2ImgPipeline
import time
import copy
import numpy as np
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from scipy.stats import chi2_contingency
import argparse
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.signal import correlate2d, correlate
from sklearn.metrics.pairwise import cosine_similarity

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

def chi_square_test(x):
    # Flatten the 2D noise to a 1D array
    flattened_noise = x.flatten()
    flattened_noise = np.array(flattened_noise)

    # Define expected frequencies based on a Gaussian distribution
    expected_freq, _ = np.histogram(flattened_noise, bins='auto', density=True)
    expected_freq *= len(flattened_noise)

    # Calculate observed frequencies
    observed_freq, _ = np.histogram(flattened_noise, bins='auto')

    # Perform the Chi-square test
    chi2_stat, p_value, _, _ = chi2_contingency([observed_freq, expected_freq])

    # Output results
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The noise distribution is significantly different from a Gaussian distribution.")
    else:
        print("Fail to reject the null hypothesis: The noise distribution is consistent with a Gaussian distribution.")

def plot_and_save_image(image, recon_image, latents, recon_latents):
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

def plot_wavelet_transformation(x):
    shape = x.shape

    max_lev = 3       # how many levels of decomposition to draw
    label_levels = 3  # how many levels to explicitly label on the plots

    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    
    # Calculate the global maximum absolute value across all coefficients
    global_max = np.abs(x).max()

    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(x)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                        label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT without normalization
        c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)

        # show the coefficients without normalization
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap='gray', vmin=-global_max, vmax=global_max)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()

    # Previous version : normalizes on each range
    """
    shape = x.shape

    max_lev = 3       # how many levels of decomposition to draw
    label_levels = 3  # how many levels to explicitly label on the plots

    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            # axes[1, 0].imshow(x, cmap=plt.cm.gray)
            axes[1, 0].imshow(x)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                        label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        # axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].imshow(arr)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()
    """

def plot_wt_hv(x):
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(x, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

def plot_distribution(original_latents, recon_latents, version="fourier"):
    if version == "fourier":
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # Adjust the number of columns

        original_latents = original_latents[0]
        recon_latents = recon_latents[0]
        latent_difference = original_latents - recon_latents

        all_latents = np.concatenate((original_latents, recon_latents, latent_difference))
        vmin_latents = all_latents.min()
        vmax_latents = all_latents.max()

        # Plot Original Latents
        im1 = axs[0, 0].imshow(original_latents, vmin=vmin_latents, vmax=vmax_latents)
        axs[0, 0].set_title('Original Latents')

        # Plot Reconstructed Latents
        im2 = axs[0, 1].imshow(recon_latents, vmin=vmin_latents, vmax=vmax_latents)
        axs[0, 1].set_title('Reconstructed Latents')

        # Plot Latents Difference
        im3 = axs[0, 2].imshow(latent_difference, vmin=vmin_latents, vmax=vmax_latents)
        axs[0, 2].set_title('Latents Difference')

        # Plot Fourier Transform - Original Latents
        fft_result_original = torch.fft.fftshift(torch.fft.fft2(original_latents))
        magnitude_spectrum_original = np.log(1+np.abs(fft_result_original))

        # Plot Fourier Transform - Reconstructed Latents
        fft_result_recon = torch.fft.fftshift(torch.fft.fft2(recon_latents))
        magnitude_spectrum_recon = np.log(1+np.abs(fft_result_recon))

        fft_difference = magnitude_spectrum_original - magnitude_spectrum_recon

        vmin_fft = np.min(np.concatenate([magnitude_spectrum_original, magnitude_spectrum_recon, fft_difference]))
        vmax_fft = np.max(np.concatenate([magnitude_spectrum_original, magnitude_spectrum_recon, fft_difference]))

        im4 = axs[1, 0].imshow(magnitude_spectrum_original, vmin=vmin_fft, vmax=vmax_fft)
        axs[1, 0].set_title('Fourier Transform - Original Latents')

        im5 = axs[1, 1].imshow(magnitude_spectrum_recon, vmin=vmin_fft, vmax=vmax_fft)
        axs[1, 1].set_title('Fourier Transform - Reconstructed Latents')

        # Plot FFT Difference
        im6 = axs[1, 2].imshow(fft_difference, vmin=vmin_fft, vmax=vmax_fft)
        axs[1, 2].set_title('FFT Difference')

        # Set up axes for unified colorbar for im1 to im3
        cax1 = fig.add_axes([0.93, 0.58, 0.01, 0.3])  # [x, y, width, height]

        # Add unified colorbar for im1 to im3
        cbar1 = plt.colorbar(im3, cax=cax1)
        cbar1.set_label('Colorbar Label 1')

        # Set up axes for unified colorbar for im4 to im6
        cax2 = fig.add_axes([0.93, 0.1, 0.01, 0.3])  # [x, y, width, height]

        # Add unified colorbar for im4 to im6
        cbar2 = plt.colorbar(im6, cax=cax2)
        cbar2.set_label('Colorbar Label 2')

        # Adjust layout for better visualization
        plt.show()

    elif version == "wavelet":
        original_latents = original_latents[0]
        recon_latents = recon_latents[0]

        plot_wavelet_transformation(original_latents)
        plot_wavelet_transformation(recon_latents)

        plot_wt_hv(original_latents)
        plot_wt_hv(recon_latents)

        # chi_square_test(original_latents)
        # chi_square_test(recon_latents)

    elif version == "correlation":
        original_latents = original_latents[0]
        recon_latents = recon_latents[0]

        corr = correlate2d(original_latents, recon_latents)
        y, x = np.unravel_index(np.argmax(corr), corr.shape)

        plt.imshow(corr)
        plt.show()


@torch.no_grad()
def set_and_generate_image_then_reverse(seed, prompt, inversion_prompt, randomize_seed, num_inference_steps=1, num_inversion_steps=1, guidance_scale=3.0):
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
        latents=original_array,
        input_type="images",
        # latents=latents,
        # input_type="latents",
        num_inversion_steps=num_inversion_steps, num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        verbose=True,
        use_random_initial_noise=False,
        decoder_inv_steps=10,
        forward_steps=100,
        tuning_steps=40,
        )
    
    print(f"TOT of inversion {(recon_latents - original_latents).norm()/original_latents.norm()}")

    # Visualizing noise
    original_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(original_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])
    recon_latents_visualized = insta_pipe.image_processor.postprocess(insta_pipe.vae.decode(recon_latents/insta_pipe.vae.config.scaling_factor, return_dict=False)[0])

    # visual = self.image_processor.postprocess(self.vae.decode(input.cpu()/self.vae.config.scaling_factor, return_dict=False)[0])

    original_latents_visualized = np.squeeze(original_latents_visualized)
    recon_latents_visualized = np.squeeze(recon_latents_visualized)

    # Obtain image, calculate OTO
    recon_image = recon_images[0]
    recon_array = insta_pipe.image_processor.pil_to_numpy(recon_image)
    diff = np.linalg.norm(original_array - recon_array)

    print(f"OTO of inversion {diff/np.linalg.norm(original_array)}")

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

    """
    # # Calculate the correlation coefficient matrix for each pair of columns
    # correlation_matrix1 = np.corrcoef(original_latents_first_channel, latents_comparison_first_channel, rowvar=False)
    # correlation_matrix2 = np.corrcoef(recon_latents_first_channel, latents_comparison_first_channel, rowvar=False)

    # # Extract the correlation coefficient between the two arrays
    # correlation_coefficient1 = correlation_matrix1[0, 1]
    # correlation_coefficient2 = correlation_matrix2[0, 1]

    # corr1 = correlate2d(original_latents_first_channel, latents_comparison_first_channel)
    # corr2 = correlate2d(recon_latents_first_channel, latents_comparison_first_channel)
    # corr1[63, 63] = 0
    # corr2[63, 63] = 0
    """

    """
    # Calculate correlations
    corrs = [np.array(latents_cpu[i] - original_noise_cpu[i]) for i in range(4)]
    corrs += [np.array(latents_cpu[i] - recon_noise_cpu[i]) for i in range(4)]

    # Determine min and max values
    min_val = min(np.min(corr) for corr in corrs)
    max_val = max(np.max(corr) for corr in corrs)

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))

    # Plot correlations
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(corrs[i], cmap='viridis', vmin=min_val, vmax=max_val)
        ax.set_title(f'Corr{i + 1}')

    # Add a single colorbar for all subplots
    cax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([min_val, max_val])

    # Adjust layout
    fig.tight_layout()

    # Show the plots
    plt.show()
    """

    # Calculate pixelwise similarity
    cosinesim_orig = []
    cosinesim_recon = []

    for i in range(64):
        for j in range(64):
            val_orig = cosine_similarity([latents_cpu[:, i, j]], [original_noise_cpu[:, i, j]])
            val_recon = cosine_similarity([latents_cpu[:, i, j]], [recon_noise_cpu[:, i, j]])
            cosinesim_orig.append(val_orig)
            cosinesim_recon.append(val_recon)

    cosinesim_orig = np.array(cosinesim_orig).reshape(64, 64)
    cosinesim_recon = np.array(cosinesim_recon).reshape(64, 64)

    print(f"original cosine sim : mean {np.abs(cosinesim_orig).mean():.3f}, max {cosinesim_orig.max():.3f}")
    print(f"recon cosine sim : mean {np.abs(cosinesim_recon).mean():.3f}, max {cosinesim_recon.max():.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axs[0].imshow(cosinesim_orig)
    axs[0].set_title('Cosine similarity (latents & origianl noise)')    

    im2 = axs[1].imshow(cosinesim_recon)
    axs[1].set_title('Cosine similarity (latents & reconstructed noise)')

    cax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')

    plt.show()

    # Section : statistical test

    # # Shapiro-Wilk test for original_latents
    # stat_original, p_value_original = shapiro(original_latents_data)
    # print(f'Shapiro-Wilk test for Original Latents: W={stat_original:.8f}, p-value={p_value_original:.8f}')

    # # Shapiro-Wilk test for recon_latents
    # stat_recon, p_value_recon = shapiro(recon_latents_data)
    # print(f'Shapiro-Wilk test for Reconstructed Latents: W={stat_recon:.8f}, p-value={p_value_recon:.8f}')

    # Section : Check with plot distribution
    plot_distribution(original_latents.cpu()[0], recon_latents.cpu()[0], version="fourier")

    latents_plot = latents_cpu[0:3]
    latents_plot = (latents_plot - latents_plot.min())/(latents_plot.max() - latents_plot.min())
    original_noise_cpu = original_noise_cpu[0:3]
    original_noise_cpu = (original_noise_cpu - original_noise_cpu.min())/(original_noise_cpu.max() - original_noise_cpu.min())
    recon_noise_cpu = recon_noise_cpu[0:3]
    recon_noise_cpu = (recon_noise_cpu - recon_noise_cpu.min())/(recon_noise_cpu.max() - recon_noise_cpu.min())

    return original_image, recon_image, original_latents_visualized, recon_latents_visualized, inf_time, seed

def main():
    image, recon_image, latents, recon_latents, time, seed = set_and_generate_image_then_reverse(
        args.seed, args.prompt, args.inversion_prompt, args.randomize_seed, 
        num_inference_steps=1, num_inversion_steps=1,
        guidance_scale=1.0
    )

    plot_and_save_image(image, recon_image, latents, recon_latents)

    print(f"generation time : {time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='instaflow - work on inversion')
    parser.add_argument('--randomize_seed', default=False, type=bool)
    parser.add_argument('--seed', default=3993109412, type=int)
    parser.add_argument('--prompt', default="Generate an image of a calm lakeside scene at sunset. Showcase vibrant wildflowers in the foreground, and distant mountains painted in the soft hues of twilight. Capture the serene beauty of nature in this tranquil", type=str)
    parser.add_argument('--inversion_prompt', default="Create a serene lakeside sunset scene with vivid wildflowers in the foreground and distant mountains adorned in the gentle twilight palette. Bring to life the tranquility and beauty of nature in this peaceful image.", type=str)
    #parser.add_argument('--prompt', default="A high resolution photo of an yellow porsche under sunshine", type=str)

    args = parser.parse_args()

    main()