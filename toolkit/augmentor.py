"""
Image Augmentation Module
Provides various augmentation transforms for expanding CV training datasets.
Each function takes an image (numpy array) and returns the augmented version
along with metadata describing the transform applied.
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Dict, List, Any


class ImageAugmentor:
    """
    Applies configurable augmentation transforms to images for dataset expansion.
    
    Supported transforms:
        - rotate: Random rotation (90, 180, 270) or arbitrary angle
        - flip_h: Horizontal flip
        - flip_v: Vertical flip
        - brightness: Random brightness adjustment
        - contrast: Random contrast adjustment
        - gaussian_noise: Additive Gaussian noise
        - salt_pepper: Salt-and-pepper noise
        - gaussian_blur: Gaussian blur
        - motion_blur: Simulated motion blur
        - crop: Random crop and resize
    """

    AVAILABLE_TRANSFORMS = [
        "rotate", "flip_h", "flip_v", "brightness", "contrast",
        "gaussian_noise", "salt_pepper", "gaussian_blur", "motion_blur", "crop"
    ]

    def __init__(self, seed: int = None):
        """
        Initialize augmentor with optional random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible augmentations. None for random.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def rotate(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Rotate image by a random angle from [90, 180, 270] degrees.
        
        Args:
            image: Input image as numpy array (BGR format).
            
        Returns:
            Tuple of (augmented_image, metadata_dict).
        """
        angle = random.choice([90, 180, 270])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                  borderMode=cv2.BORDER_REFLECT_101)
        metadata = {"transform": "rotate", "angle": angle}
        return rotated, metadata

    def flip_h(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Flip image horizontally.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (flipped_image, metadata_dict).
        """
        flipped = cv2.flip(image, 1)
        return flipped, {"transform": "flip_h"}

    def flip_v(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Flip image vertically.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (flipped_image, metadata_dict).
        """
        flipped = cv2.flip(image, 0)
        return flipped, {"transform": "flip_v"}

    def brightness(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly adjust image brightness by scaling pixel values.
        Factor range: [0.5, 1.5] (0.5 = darker, 1.5 = brighter).
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (adjusted_image, metadata_dict).
        """
        factor = round(random.uniform(0.5, 1.5), 2)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted, {"transform": "brightness", "factor": factor}

    def contrast(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly adjust image contrast using linear scaling around the mean.
        Factor range: [0.5, 1.8].
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (adjusted_image, metadata_dict).
        """
        factor = round(random.uniform(0.5, 1.8), 2)
        mean = np.mean(image)
        adjusted = np.clip((image.astype(np.float64) - mean) * factor + mean, 0, 255)
        return adjusted.astype(np.uint8), {"transform": "contrast", "factor": factor}

    def gaussian_noise(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Add Gaussian (random) noise to simulate sensor noise.
        Standard deviation randomly chosen from [10, 50].
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (noisy_image, metadata_dict).
        """
        sigma = random.randint(10, 50)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
        noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return noisy, {"transform": "gaussian_noise", "sigma": sigma}

    def salt_pepper(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Add salt-and-pepper noise (random white and black pixels).
        Noise density randomly chosen from [0.01, 0.05].
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (noisy_image, metadata_dict).
        """
        density = round(random.uniform(0.01, 0.05), 3)
        noisy = image.copy()
        total_pixels = image.shape[0] * image.shape[1]

        # Salt (white pixels)
        num_salt = int(total_pixels * density)
        salt_coords = [
            np.random.randint(0, image.shape[0], num_salt),
            np.random.randint(0, image.shape[1], num_salt)
        ]
        noisy[salt_coords[0], salt_coords[1]] = 255

        # Pepper (black pixels)
        num_pepper = int(total_pixels * density)
        pepper_coords = [
            np.random.randint(0, image.shape[0], num_pepper),
            np.random.randint(0, image.shape[1], num_pepper)
        ]
        noisy[pepper_coords[0], pepper_coords[1]] = 0

        return noisy, {"transform": "salt_pepper", "density": density}

    def gaussian_blur(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Gaussian blur to simulate out-of-focus conditions.
        Kernel size randomly chosen from [3, 5, 7, 9].
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (blurred_image, metadata_dict).
        """
        ksize = random.choice([3, 5, 7, 9])
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred, {"transform": "gaussian_blur", "kernel_size": ksize}

    def motion_blur(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulate motion blur using a directional kernel.
        Kernel size randomly chosen from [5, 10, 15, 20].
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (blurred_image, metadata_dict).
        """
        ksize = random.choice([5, 10, 15, 20])
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = np.ones(ksize) / ksize
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred, {"transform": "motion_blur", "kernel_size": ksize}

    def crop(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Random crop between 60-90% of original size, then resize back.
        Simulates different framing and zoom levels.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (cropped_image, metadata_dict).
        """
        h, w = image.shape[:2]
        crop_ratio = round(random.uniform(0.6, 0.9), 2)
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        cropped = image[top:top + new_h, left:left + new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        metadata = {
            "transform": "crop",
            "crop_ratio": crop_ratio,
            "region": {"top": top, "left": left, "height": new_h, "width": new_w}
        }
        return resized, metadata

    def apply_random(self, image: np.ndarray,
                     transforms: List[str] = None,
                     count: int = 1) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply random augmentations to an image.
        
        Args:
            image: Input image as numpy array.
            transforms: List of transform names to choose from.
                        If None, uses all available transforms.
            count: Number of augmented variants to produce.
            
        Returns:
            List of (augmented_image, metadata_dict) tuples.
        """
        if transforms is None:
            transforms = self.AVAILABLE_TRANSFORMS
        else:
            # Validate requested transforms
            invalid = set(transforms) - set(self.AVAILABLE_TRANSFORMS)
            if invalid:
                raise ValueError(
                    f"Unknown transforms: {invalid}. "
                    f"Available: {self.AVAILABLE_TRANSFORMS}"
                )

        results = []
        for _ in range(count):
            transform_name = random.choice(transforms)
            transform_fn = getattr(self, transform_name)
            augmented, metadata = transform_fn(image)
            results.append((augmented, metadata))

        return results

    def apply_all(self, image: np.ndarray,
                  transforms: List[str] = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply all specified transforms to an image (one variant per transform).
        
        Args:
            image: Input image as numpy array.
            transforms: List of transform names. If None, applies all.
            
        Returns:
            List of (augmented_image, metadata_dict) tuples.
        """
        if transforms is None:
            transforms = self.AVAILABLE_TRANSFORMS

        results = []
        for transform_name in transforms:
            transform_fn = getattr(self, transform_name)
            augmented, metadata = transform_fn(image)
            results.append((augmented, metadata))

        return results
