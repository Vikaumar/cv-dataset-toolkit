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
from scipy.ndimage import map_coordinates, gaussian_filter


class ImageAugmentor:
    """
    Applies configurable augmentation transforms to images for dataset expansion.

    Supported transforms (16 total):
        Geometric:
            - rotate: Random rotation (90, 180, 270 degrees)
            - flip_h: Horizontal flip
            - flip_v: Vertical flip
            - crop: Random crop and resize
            - perspective: Random perspective warp
            - elastic: Elastic deformation

        Color / Intensity:
            - brightness: Random brightness adjustment
            - contrast: Random contrast adjustment
            - clahe: Contrast-Limited Adaptive Histogram Equalization
            - hue_shift: Random hue channel shift
            - channel_shuffle: Random BGR channel reordering

        Noise / Degradation:
            - gaussian_noise: Additive Gaussian noise
            - salt_pepper: Salt-and-pepper noise
            - gaussian_blur: Gaussian blur
            - motion_blur: Simulated motion blur
            - cutout: Random rectangular region erasure
    """

    AVAILABLE_TRANSFORMS = [
        "rotate", "flip_h", "flip_v", "brightness", "contrast",
        "gaussian_noise", "salt_pepper", "gaussian_blur", "motion_blur", "crop",
        "perspective", "elastic", "clahe", "hue_shift", "channel_shuffle", "cutout"
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

    # ---- New Transforms (Phase 1) ----

    def perspective(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply random perspective warp to simulate camera angle changes.

        Randomly displaces the four corner points by up to 10% of the
        image dimensions, then computes a perspective transform matrix.

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (warped_image, metadata_dict).
        """
        h, w = image.shape[:2]
        margin = 0.1  # max displacement as fraction of dimension

        # Original corner points
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Randomly displaced destination points
        dx = int(w * margin)
        dy = int(h * margin)
        dst = np.float32([
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (w, h),
                                      borderMode=cv2.BORDER_REFLECT_101)

        metadata = {
            "transform": "perspective",
            "margin": margin,
            "dst_points": dst.tolist()
        }
        return warped, metadata

    def elastic(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply elastic deformation to the image.

        Generates random displacement fields, smooths them with a Gaussian
        filter, then applies the deformation. Commonly used for augmenting
        medical imaging and document OCR datasets.

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (deformed_image, metadata_dict).
        """
        alpha = random.uniform(80, 120)   # displacement intensity
        sigma = random.uniform(9, 11)     # smoothing factor
        alpha = round(alpha, 1)
        sigma = round(sigma, 1)

        h, w = image.shape[:2]
        random_state = np.random.RandomState(None)

        # Generate smooth random displacement fields
        dx = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter(
            (random_state.rand(h, w) * 2 - 1), sigma) * alpha

        # Create coordinate grids
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        deformed = cv2.remap(image, map_x, map_y,
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

        metadata = {
            "transform": "elastic",
            "alpha": alpha,
            "sigma": sigma
        }
        return deformed, metadata

    def clahe(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Enhances local contrast while limiting noise amplification.
        Applied to the L channel in LAB color space to preserve colors.
        Clip limit randomly chosen from [1.0, 4.0].
        Grid size randomly chosen from [4, 8, 16].

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (enhanced_image, metadata_dict).
        """
        clip_limit = round(random.uniform(1.0, 4.0), 1)
        grid_size = random.choice([4, 8, 16])

        # Convert to LAB, apply CLAHE to L channel only
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe_op = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )
        l_enhanced = clahe_op.apply(l_channel)

        enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        metadata = {
            "transform": "clahe",
            "clip_limit": clip_limit,
            "grid_size": grid_size
        }
        return enhanced, metadata

    def hue_shift(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly shift the hue channel in HSV color space.

        Shift range: [-30, +30] on the 0-180 OpenCV hue scale.
        Simulates different lighting color temperatures.

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (shifted_image, metadata_dict).
        """
        shift = random.randint(-30, 30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return shifted, {"transform": "hue_shift", "shift": shift}

    def channel_shuffle(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomly reorder the BGR color channels.

        Produces color-shifted variants that force models to learn
        shape features rather than relying on color cues.

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (shuffled_image, metadata_dict).
        """
        channels = list(range(3))
        random.shuffle(channels)
        shuffled = image[:, :, channels]

        channel_names = ["B", "G", "R"]
        order = [channel_names[c] for c in channels]
        return shuffled, {"transform": "channel_shuffle", "order": order}

    def cutout(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Erase a random rectangular region (filled with zeros/black).

        Simulates occlusion and forces models to learn from partial views.
        Region covers 5-20% of image area. 1-3 rectangles are erased.

        Args:
            image: Input image as numpy array.

        Returns:
            Tuple of (cutout_image, metadata_dict).
        """
        result = image.copy()
        h, w = image.shape[:2]
        num_rects = random.randint(1, 3)
        regions = []

        for _ in range(num_rects):
            # Rectangle covers 5-20% of one dimension
            rect_h = random.randint(int(h * 0.05), int(h * 0.2))
            rect_w = random.randint(int(w * 0.05), int(w * 0.2))
            top = random.randint(0, h - rect_h)
            left = random.randint(0, w - rect_w)

            result[top:top + rect_h, left:left + rect_w] = 0
            regions.append({
                "top": top, "left": left,
                "height": rect_h, "width": rect_w
            })

        metadata = {
            "transform": "cutout",
            "num_rects": num_rects,
            "regions": regions
        }
        return result, metadata

    # ---- Apply Methods ----

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

