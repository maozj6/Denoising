import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.signal import wiener
from skimage import color, data, util
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


from skimage.restoration import richardson_lucy

def lucy_richardson_denoise(image, psf, iterations):
    """
    Perform Lucy-Richardson deconvolution.
    """
    image = image.astype(np.float64) / 255  # Normalize image
    psf = psf / psf.sum()  # Normalize PSF
    deblurred = richardson_lucy(image, psf, iterations)
    return (deblurred * 255).astype(np.uint8)

def add_brightness_contrast(image):
    brightness = 50
    contrast = 1.5
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def add_gaussian_noise(image):
    gaussian_noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, gaussian_noise)

def add_pepper_noise(image):
    pepper_noise = util.random_noise(image, mode='pepper', amount=0.02)
    return (pepper_noise * 255).astype(np.uint8)

def add_motion_blur(image):
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(image, -1, kernel)

def variational_bayesian_denoise(image, n_components=2):
    """
    Apply Variational Bayesian Inference for denoising using Gaussian Mixture Models.
    """
    image = image.astype(np.float64) / 255  # Normalize image
    flat_image = image.flatten().reshape(-1, 1)

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=100)
    gmm.fit(flat_image)

    # Assign each pixel to its most probable Gaussian component
    denoised_image = gmm.means_[gmm.predict(flat_image)].reshape(image.shape)
    return (denoised_image * 255).astype(np.uint8)

def frequency_inverse_filtering(image, psf):
    """
    Apply frequency domain inverse filtering for deblurring.
    """
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)

    # Avoid division by zero
    psf_fft[np.abs(psf_fft) < 1e-8] = 1e-8

    deblurred_fft = image_fft / psf_fft
    deblurred_image = np.fft.ifft2(deblurred_fft)
    return np.abs(deblurred_image)

def l2_regularized_deconvolution(image, psf, alpha):
    """
    Perform L2-regularized deconvolution.
    """
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(psf, s=image.shape)
    psf_conj_fft = np.conj(psf_fft)

    # Regularization in the frequency domain
    denominator = psf_conj_fft * psf_fft + alpha
    deblurred_fft = (psf_conj_fft * image_fft) / denominator
    deblurred_image = np.fft.ifft2(deblurred_fft)
    return np.abs(deblurred_image)

# Load Lena image
data_image = data.camera()  # Use Lena image from skimage
original_image = util.img_as_ubyte(data_image)

# Add noise
noisy_images = {
    "Brightness/Contrast": add_brightness_contrast(original_image),
    "Gaussian Noise": add_gaussian_noise(original_image),
    "Pepper Noise": add_pepper_noise(original_image),
    "Motion Blur": add_motion_blur(original_image)
}

# Define PSF (Point Spread Function, e.g., Gaussian blur kernel)
def gaussian_psf(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

psf = gaussian_psf(15, 3)

# Apply denoising and deblurring
denoised_images = {}
alpha = 0.01  # Regularization parameter
for noise_type, noisy_image in noisy_images.items():
    if noise_type == "Brightness/Contrast":
        denoised_images[noise_type] = noisy_image  # Skip denoising for brightness/contrast
    elif noise_type == "Motion Blur":
        denoised_images[noise_type] = l2_regularized_deconvolution(noisy_image, psf, alpha)
    else:
        denoised_images[noise_type] = variational_bayesian_denoise(noisy_image)

# Plot results
fig, axes = plt.subplots(len(noisy_images), 3, figsize=(15, 5 * len(noisy_images)))

for i, (noise_type, noisy_image) in enumerate(noisy_images.items()):
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(noisy_image, cmap='gray')
    axes[i, 1].set_title(f"Noisy Image ({noise_type})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(denoised_images[noise_type], cmap='gray')
    axes[i, 2].set_title("Denoised Image")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
