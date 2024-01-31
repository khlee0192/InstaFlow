from scipy.stats import chi2_contingency
from scipy.signal import correlate2d, correlate
from datetime import datetime

import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np
import matplotlib.pyplot as plt

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

def plot_and_save_image(image, recon_image, latents, recon_latents, show):
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

    if show:
        plt.show()
    else:
        plt.savefig(f"plot_{current_time}.png")

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

def plot_distribution(original_latents, recon_latents, latents=None, version="fourier"):
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

    elif version == "cosinesim":
            # Calculate pixelwise similarity
        cosinesim_orig = []
        cosinesim_recon = []

        for i in range(64):
            for j in range(64):
                val_orig = cosine_similarity([latents[:, i, j]], [original_latents[:, i, j]])
                val_recon = cosine_similarity([latents[:, i, j]], [recon_latents[:, i, j]])
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