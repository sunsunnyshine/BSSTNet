import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set the font properties to a font that supports Chinese characters
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Path to a Chinese font file
font_prop = fm.FontProperties(fname=font_path)

class BlurDetector:
    SENSITIVITY_PRESETS = {
        'very_low': {'sigma': 12.0, 'block_sizes': [16, 32, 64]},
        'low': {'sigma': 10.0, 'block_sizes': [8, 16, 32]},
        'normal': {'sigma': 8.0, 'block_sizes': [4, 8, 16]},
        'high': {'sigma': 6.0, 'block_sizes': [2, 4, 8]},
        # 'very_high': {'sigma': 4.0, 'block_sizes': [2, 3, 4]}
    }

    def __init__(self,
                 device: Optional[str] = None,
                 sensitivity: str = 'normal',
                 custom_sigma: Optional[float] = None,
                 custom_block_sizes: Optional[List[int]] = None,
                 computation_stride: int = 16):  # New parameter for computation stride
        """
        Initialize blur detector with configurable sensitivity and computation stride
        Args:
            device: 'cuda' for GPU, 'cpu' for CPU, None for automatic selection
            sensitivity: 'low', 'normal', 'high', or 'very_high'
            custom_sigma: Optional custom sigma value for reblur
            custom_block_sizes: Optional custom block sizes for analysis
            computation_stride: Stride for block-wise computation (higher = faster but less precise)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.computation_stride = computation_stride
        print(f"Using device: {self.device}")

        if custom_sigma is not None or custom_block_sizes is not None:
            self.sigma = custom_sigma or 2.5
            self.block_sizes = custom_block_sizes or [4, 8, 16]
        else:
            config = self.SENSITIVITY_PRESETS[sensitivity]
            self.sigma = config['sigma']
            self.block_sizes = config['block_sizes']

        print(f"Configuration: sigma={self.sigma}, block_sizes={self.block_sizes}, stride={computation_stride}")
        self.dct_matrices = {}

    def _get_dct_matrix(self, size: int) -> torch.Tensor:
        """Get or create DCT matrix for given size"""
        if size not in self.dct_matrices:
            matrix = torch.zeros((size, size), dtype=torch.float32, device=self.device)
            for i in range(size):
                for j in range(size):
                    if i == 0:
                        matrix[i, j] = 1 / np.sqrt(size)
                    else:
                        matrix[i, j] = np.sqrt(2 / size) * np.cos(np.pi * (2 * j + 1) * i / (2 * size))
            self.dct_matrices[size] = matrix
        return self.dct_matrices[size]

    def _dct2d(self, x: torch.Tensor, dct_m: torch.Tensor) -> torch.Tensor:
        """Compute 2D DCT using matrix multiplication"""
        return torch.matmul(torch.matmul(dct_m, x), dct_m.T)

    def _extract_blocks_strided(self, img: torch.Tensor, block_size: int, stride: int) -> Tuple[
        torch.Tensor, Tuple[int, int]]:
        """Extract blocks with stride for efficient computation"""
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)

        # Extract patches with stride
        patches = F.unfold(img,
                           kernel_size=(block_size, block_size),
                           stride=stride)

        # Calculate output dimensions
        height = (img.shape[2] - block_size) // stride + 1
        width = (img.shape[3] - block_size) // stride + 1

        # Reshape to [num_blocks, block_size, block_size]
        patches = patches.transpose(1, 2).reshape(-1, block_size, block_size)

        return patches, (height, width)


    def detect_blur(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Detect blur in image using block-wise computation and upsampling
        Args:
            img_tensor: Input tensor of shape [B, C, H, W]
        Returns:
            Blur map tensor of shape [B, 1, H, W]
        """
        # Ensure input is on the correct device
        img_tensor = img_tensor.to(self.device)

        # Get original dimensions
        batch_size, channels, original_height, original_width = img_tensor.shape

        # Convert to grayscale if necessary
        if channels > 1:
            img_tensor = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
        else:
            img_tensor = img_tensor.squeeze(1)

        # Apply reblur
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        kernel = cv2.getGaussianKernel(kernel_size, self.sigma).astype(np.float32)
        kernel = torch.from_numpy(np.outer(kernel, kernel)).to(self.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()

        # Apply blur using conv2d
        reblur_tensor = F.conv2d(img_tensor.unsqueeze(1), kernel, padding=kernel_size // 2)
        reblur_tensor = reblur_tensor.squeeze(1)

        # Process each block size
        blur_maps = []
        for block_size in self.block_sizes:
            # Extract blocks with stride
            blocks_orig, (height, width) = self._extract_blocks_strided(
                img_tensor.unsqueeze(1), block_size, self.computation_stride)
            blocks_reblur, _ = self._extract_blocks_strided(
                reblur_tensor.unsqueeze(1),
                block_size,
                self.computation_stride)

            # Get DCT matrix
            dct_m = self._get_dct_matrix(block_size)

            # Compute DCT for all blocks in parallel
            dct_blocks = torch.stack([self._dct2d(b, dct_m) for b in blocks_orig])
            dct_reblur = torch.stack([self._dct2d(b, dct_m) for b in blocks_reblur])

            # Compute L1 norm ratio
            norm_orig = torch.sum(torch.abs(dct_blocks), dim=(1, 2))
            norm_reblur = torch.sum(torch.abs(dct_reblur), dim=(1, 2))

            # Compute beta values
            beta = torch.where(norm_orig > 0,
                               torch.clamp(norm_reblur / norm_orig, 0, 1),
                               torch.zeros_like(norm_orig))

            # Reshape beta to match the strided output dimensions
            blur_map = beta.reshape(batch_size, height, width)

            # Upsample to original size using bilinear interpolation
            blur_map = F.interpolate(
                blur_map.unsqueeze(1),
                size=(original_height, original_width),
                mode='bilinear',
                align_corners=False
            )

            blur_maps.append(blur_map)

        # Average all blur maps
        final_blur_map = torch.stack(blur_maps).mean(0)

        return final_blur_map


def visualize_blur_map(blur_map: np.ndarray) -> np.ndarray:
    """Convert blur map to visualization"""
    vis_map = (blur_map * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)


def main_old():
    # Example usage with different stride values
    test_tensor = torch.Tensor((1, 3, 256, 256))

    try:
        # Test different computation strides
        strides = [8, 16, 32]
        for stride in strides:
            detector = BlurDetector(
                sensitivity='very_low',
                computation_stride=stride
            )

            # Time the computation
            import time
            start_time = time.time()

            blur_map = detector.detect_blur(test_tensor)

            elapsed_time = time.time() - start_time
            print(f"Processing time with stride {stride}: {elapsed_time:.2f} seconds")

            # Save results
            cv2.imwrite(f"blur_map_stride_{stride}.png", (blur_map * 255).astype(np.uint8))
            # save the sharp mask
            cv2.imwrite(f"sharp_map_stride_{stride}.png", ((1 - blur_map) * 255).astype(np.uint8))

        print("Processing completed. Results saved for all stride values.")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Create a batch of random images
    batch_size = 4
    channels = 3
    height = 256
    width = 256

    # Generate random images
    images = torch.rand(batch_size, channels, height, width)

    # Add some blur to half of the images
    blurred_images = images.clone()
    kernel_size = 15
    kernel = torch.ones(kernel_size, kernel_size) / (kernel_size * kernel_size)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    blurred_images[batch_size // 2:] = torch.nn.functional.conv2d(
        blurred_images[batch_size // 2:], kernel, padding=kernel_size // 2, groups=channels
    )

    # Initialize BlurDetector
    detector = BlurDetector(sensitivity='very_low', computation_stride=16)

    # Detect blur
    blur_maps = detector.detect_blur(blurred_images)

    # Visualize results
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    for i in range(batch_size):
        # Display original image
        axes[i, 0].imshow(blurred_images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        # Display blur map
        im = axes[i, 1].imshow(blur_maps[i, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Blur Map {i + 1}")
        axes[i, 1].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig('blur_detection_results.png')
    print("Results saved as 'blur_detection_results.png'")


def load_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def test_parameters(image_path):
    # Load image
    img_tensor = load_image(image_path)

    # Crop the image tensor (example: crop to a 300x300 region starting from (100, 100))
    # img_tensor = img_tensor[:, :, 100:400, 100:400]

    # Define parameter combinations to test
    sensitivities = ['very_low', 'low', 'normal', 'high']
    strides = [8, 16, 32]


    # Create a figure to display results
    fig, axes = plt.subplots(len(sensitivities), len(strides) + 1)

    for i, sensitivity in enumerate(sensitivities):
        for j, stride in enumerate(strides):
            # Initialize BlurDetector with current parameters
            detector = BlurDetector(sensitivity=sensitivity, computation_stride=stride)

            # Detect blur
            import time
            start_time = time.time()

            blur_map = detector.detect_blur(img_tensor)

            elapsed_time = time.time() - start_time
            print(f"Processing time with stride {stride}: {elapsed_time:.2f} seconds")
            sharp_map = 1 - blur_map
            # Plot result
            ax = axes[i, j + 1]
            im = ax.imshow(blur_map[0, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            # ax.set_title(f"再模糊系数: {BlurDetector.SENSITIVITY_PRESETS[sensitivity]}  窗口大小: {stride}"),font_prop=font_prop)
            ax.axis('off')
            # plt.colorbar(im, ax=ax)

        # Plot the original image in the first column
        ax = axes[i, 0]
        im = ax.imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        # ax.set_title("Original Image")
        ax.axis('off')
        # plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('blur_detection_parameter_test.png')
    print("Results saved as 'blur_detection_parameter_test.png'")

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

def blur_image(image, kernel_size, sigma):
    """Apply Gaussian blur to an image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return cv2.filter2D(image, -1, kernel)

def mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)


def visualize_fig2():
    # Load the image
    img = cv2.imread('/media/yjh/yjh/LightReBlur/VDV2/scene29/00000.jpg', cv2.IMREAD_GRAYSCALE)
    # crop the image
    img = img[100:400, 100:400]


import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def blur_image(image, kernel_size, sigma):
    """Apply Gaussian blur to an RGB image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return cv2.filter2D(image, -1, kernel)


def mse(img1, img2):
    """Calculate Mean Squared Error between two RGB images."""
    return np.mean((img1 - img2) ** 2)

def visualize_fig2():
    # Load the RGB image
    img = cv2.imread('/media/yjh/yjh/LightReBlur/VDV2/scene29/00000.jpg')
    # crop the image
    img = img[100:400, 100:400]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create blurred versions
    blur1 = blur_image(img, 15, 2)
    blur2 = blur_image(img, 15, 4)

    # Create re-blurred versions
    reblur_orig = blur_image(img, 15, 0.5)
    reblur1 = blur_image(blur1, 15, 0.5)
    reblur2 = blur_image(blur2, 15, 0.5)

    # Calculate MSE
    mse_orig = mse(img, reblur_orig)
    mse_blur1 = mse(blur1, reblur1)
    mse_blur2 = mse(blur2, reblur2)

    # Display results
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original')
    axs[0, 1].imshow(blur1)
    axs[0, 1].set_title('Blur (σ=2)')
    axs[0, 2].imshow(blur2)
    axs[0, 2].set_title('Blur (σ=4)')
    axs[1, 0].imshow(reblur_orig)
    axs[1, 0].set_title(f'Re-blur (MSE={mse_orig:.4f})')
    axs[1, 1].imshow(reblur1)
    axs[1, 1].set_title(f'Re-blur (MSE={mse_blur1:.4f})')
    axs[1, 2].imshow(reblur2)
    axs[1, 2].set_title(f'Re-blur (MSE={mse_blur2:.4f})')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_fig3():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def gaussian_kernel(size, sigma):
        """Generate a 2D Gaussian kernel."""
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    def blur_image(image, kernel_size, sigma):
        """Apply Gaussian blur to an RGB image."""
        kernel = gaussian_kernel(kernel_size, sigma)
        return cv2.filter2D(image, -1, kernel)

    def dct2(a):
        """Compute 2D DCT of image."""
        return cv2.dct(np.float32(a))

    def visualize_dct(dct_coef):
        """Visualize DCT coefficients."""
        dct_log = np.log(np.abs(dct_coef) + 1)
        return (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))

    img = cv2.imread('/media/yjh/yjh/LightReBlur/VDV2/scene29/00000.jpg')
    # crop the image
    img = img[100:400, 100:400]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create blurred and re-blurred versions
    blur = blur_image(img, 15, 1)
    reblur = blur_image(blur, 15, 1)

    # Compute DCT for each channel
    dct_orig = np.array([dct2(img[:, :, i]) for i in range(3)])
    dct_blur = np.array([dct2(blur[:, :, i]) for i in range(3)])
    dct_reblur = np.array([dct2(reblur[:, :, i]) for i in range(3)])

    # Visualize DCT (average across channels)
    vis_orig = np.mean([visualize_dct(dct_orig[i]) for i in range(3)], axis=0)
    vis_blur = np.mean([visualize_dct(dct_blur[i]) for i in range(3)], axis=0)
    vis_reblur = np.mean([visualize_dct(dct_reblur[i]) for i in range(3)], axis=0)

    # Display results
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(blur)
    axs[0, 1].set_title('Blurred Image')
    axs[0, 2].imshow(reblur)
    axs[0, 2].set_title('Re-blurred Image')

    axs[1, 0].imshow(vis_orig, cmap='gray')
    axs[1, 0].set_title('DCT of Original')
    axs[1, 1].imshow(vis_blur, cmap='gray')
    axs[1, 1].set_title('DCT of Blurred')
    axs[1, 2].imshow(vis_reblur, cmap='gray')
    axs[1, 2].set_title('DCT of Re-blurred')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Compute and print the percentage of non-zero DCT coefficients
    threshold = 1e-6  # Threshold to consider a coefficient as non-zero

    def count_nonzero(dct):
        return np.mean([np.sum(np.abs(dct[i]) > threshold) / dct[i].size * 100 for i in range(3)])

    print(f"Percentage of non-zero DCT coefficients (averaged across channels):")
    print(f"Original: {count_nonzero(dct_orig):.2f}%")
    print(f"Blurred: {count_nonzero(dct_blur):.2f}%")
    print(f"Re-blurred: {count_nonzero(dct_reblur):.2f}%")
def demo():
    blurred_images = load_image(image_path)
    detector = BlurDetector(sensitivity='low', computation_stride=8)

    # Detect blur
    blur_maps = detector.detect_blur(blurred_images)
    sharp_map = 1 - blur_maps
    import cv2

    # Apply the 'jet' colormap
    sharp_map_np = sharp_map[0, 0].cpu().numpy()
    sharp_map_colored = cv2.applyColorMap((sharp_map_np * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Save the image
    cv2.imwrite('sharp_map_jet.png', sharp_map_colored)

if __name__ == "__main__":
    image_path = "/media/yjh/yjh/LightReBlur/VDV/test/blur/scene33/00005.png"  # Replace with your image path
    test_parameters(image_path)
    # demo()
    # visualize_fig2()
    # visualize_fig3()

