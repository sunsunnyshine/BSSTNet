import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class BlurDetector:
    SENSITIVITY_PRESETS = {
        'very_low': {'sigma': 12.0, 'block_sizes': [16, 32, 64]},
        'low': {'sigma': 10.0, 'block_sizes': [8, 16, 32]},
        'normal': {'sigma': 8.0, 'block_sizes': [4, 8, 16]},
        'high': {'sigma': 6.0, 'block_sizes': [2, 4, 8]},
        'very_high': {'sigma': 4.0, 'block_sizes': [2, 3, 4]}
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

    # Define parameter combinations to test
    sensitivities = ['very_low', 'low', 'normal', 'high', 'very_high']
    strides = [8, 16, 32]


    # Create a figure to display results
    fig, axes = plt.subplots(len(sensitivities), len(strides) + 1, figsize=(5 * (len(strides) + 1), 5 * len(sensitivities)))

    for i, sensitivity in enumerate(sensitivities):
        for j, stride in enumerate(strides):
            # Initialize BlurDetector with current parameters
            detector = BlurDetector(sensitivity=sensitivity, computation_stride=stride)

            # Detect blur
            blur_map = detector.detect_blur(img_tensor)
            sharp_map = 1 - blur_map
            # Plot result
            ax = axes[i, j + 1]
            im = ax.imshow(sharp_map[0, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            ax.set_title(f"Sensitivity: {sensitivity}\nStride: {stride}")
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Plot the original image in the first column
        ax = axes[i, 0]
        im = ax.imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        ax.set_title("Original Image")
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('blur_detection_parameter_test.png')
    print("Results saved as 'blur_detection_parameter_test.png'")


if __name__ == "__main__":
    image_path = "test.png"  # Replace with your image path
    test_parameters(image_path)

