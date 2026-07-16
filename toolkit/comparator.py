"""
Image Comparison Module
Provides tools for comparing images and detecting duplicates in datasets.

Features:
    - SSIM (Structural Similarity Index) comparison
    - Histogram-based color distribution comparison
    - Perceptual hashing (dHash) for near-duplicate detection
    - Batch duplicate finder for dataset deduplication
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ComparisonResult:
    """
    Stores the result of comparing two images.

    Attributes:
        image_a: Filename of the first image.
        image_b: Filename of the second image.
        ssim: Structural Similarity Index (0-1, 1 = identical).
        histogram_similarity: Histogram correlation (-1 to 1, 1 = identical).
        hash_distance: Hamming distance between perceptual hashes (0 = identical).
        is_duplicate: Whether the images are considered duplicates.
    """
    image_a: str = ""
    image_b: str = ""
    ssim: float = 0.0
    histogram_similarity: float = 0.0
    hash_distance: int = 0
    is_duplicate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "image_a": self.image_a,
            "image_b": self.image_b,
            "ssim": self.ssim,
            "histogram_similarity": self.histogram_similarity,
            "hash_distance": self.hash_distance,
            "is_duplicate": self.is_duplicate
        }


class ImageComparator:
    """
    Compares images using multiple similarity metrics and detects duplicates.
    """

    def __init__(self, ssim_threshold: float = 0.95,
                 hash_threshold: int = 5):
        """
        Initialize comparator with duplicate detection thresholds.

        Args:
            ssim_threshold: SSIM above this value = duplicate (default: 0.95).
            hash_threshold: Hash distance below this = duplicate (default: 5).
        """
        self.ssim_threshold = ssim_threshold
        self.hash_threshold = hash_threshold

    def compute_ssim(self, image_a: np.ndarray,
                     image_b: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two images.

        SSIM measures perceived quality by comparing luminance, contrast,
        and structure. Returns a value between 0 and 1, where 1 means
        the images are identical.

        Images are resized to a common dimension if they differ in size.

        Args:
            image_a: First image as numpy array.
            image_b: Second image as numpy array.

        Returns:
            SSIM value as float (0 to 1).
        """
        # Convert to grayscale
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        # Resize to common dimensions if needed
        if gray_a.shape != gray_b.shape:
            h = min(gray_a.shape[0], gray_b.shape[0])
            w = min(gray_a.shape[1], gray_b.shape[1])
            gray_a = cv2.resize(gray_a, (w, h))
            gray_b = cv2.resize(gray_b, (w, h))

        # SSIM computation constants (Wang et al., 2004)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Compute means and variances using Gaussian window
        mu_a = cv2.GaussianBlur(gray_a.astype(np.float64), (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(gray_b.astype(np.float64), (11, 11), 1.5)

        mu_a_sq = mu_a ** 2
        mu_b_sq = mu_b ** 2
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(
            gray_a.astype(np.float64) ** 2, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(
            gray_b.astype(np.float64) ** 2, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(
            gray_a.astype(np.float64) * gray_b.astype(np.float64),
            (11, 11), 1.5) - mu_ab

        # SSIM formula
        numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    def compute_histogram_similarity(self, image_a: np.ndarray,
                                      image_b: np.ndarray) -> float:
        """
        Compare color histograms of two images using correlation.

        Computes histograms for each color channel and averages the
        correlation scores. Returns -1 to 1, where 1 means identical
        color distributions.

        Args:
            image_a: First image as numpy array.
            image_b: Second image as numpy array.

        Returns:
            Average histogram correlation as float (-1 to 1).
        """
        scores = []
        for i in range(3):  # B, G, R channels
            hist_a = cv2.calcHist([image_a], [i], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_b], [i], None, [256], [0, 256])

            # Normalize histograms
            cv2.normalize(hist_a, hist_a)
            cv2.normalize(hist_b, hist_b)

            score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
            scores.append(score)

        return round(float(np.mean(scores)), 4)

    def compute_dhash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute difference hash (dHash) of an image.

        dHash is a perceptual hash that captures the gradient direction
        between adjacent pixels. It's fast and effective for finding
        near-duplicate images regardless of size or minor modifications.

        Args:
            image: Input image as numpy array.
            hash_size: Size of the hash grid (default: 8 produces 64-bit hash).

        Returns:
            Hash as integer.
        """
        # Resize to (hash_size + 1) x hash_size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        # Compute horizontal gradient (left pixel > right pixel)
        diff = resized[:, 1:] > resized[:, :-1]

        # Convert boolean array to integer hash
        return sum(2 ** i for i, v in enumerate(diff.flatten()) if v)

    def compute_hash_distance(self, hash_a: int, hash_b: int) -> int:
        """
        Compute Hamming distance between two hashes.

        The Hamming distance counts the number of positions where the
        corresponding bits differ. Lower distance = more similar.

        Args:
            hash_a: First hash as integer.
            hash_b: Second hash as integer.

        Returns:
            Hamming distance as integer.
        """
        xor = hash_a ^ hash_b
        return bin(xor).count('1')

    def compare(self, image_a: np.ndarray, image_b: np.ndarray,
                name_a: str = "", name_b: str = "") -> ComparisonResult:
        """
        Run full comparison between two images.

        Args:
            image_a: First image as numpy array.
            image_b: Second image as numpy array.
            name_a: Filename of first image.
            name_b: Filename of second image.

        Returns:
            ComparisonResult with all metrics.
        """
        result = ComparisonResult(image_a=name_a, image_b=name_b)

        result.ssim = round(self.compute_ssim(image_a, image_b), 4)
        result.histogram_similarity = self.compute_histogram_similarity(
            image_a, image_b)

        hash_a = self.compute_dhash(image_a)
        hash_b = self.compute_dhash(image_b)
        result.hash_distance = self.compute_hash_distance(hash_a, hash_b)

        # Determine if duplicate based on thresholds
        result.is_duplicate = result.ssim >= self.ssim_threshold

        return result

    def find_duplicates(self, images: List[Tuple[np.ndarray, str]],
                        ) -> Dict[str, Any]:
        """
        Scan a list of images and find all duplicate/near-duplicate pairs.

        Uses dHash for fast initial screening, then SSIM for confirmation.

        Args:
            images: List of (image_array, filename) tuples.

        Returns:
            Dictionary with duplicate pairs, unique count, and total comparisons.
        """
        # Pre-compute all hashes
        hashes = []
        for img, name in images:
            h = self.compute_dhash(img)
            hashes.append((h, name, img))

        duplicates = []
        total_comparisons = 0

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                total_comparisons += 1
                hash_dist = self.compute_hash_distance(
                    hashes[i][0], hashes[j][0])

                # Fast screening: only compute SSIM if hashes are close
                if hash_dist <= self.hash_threshold * 2:
                    ssim_val = self.compute_ssim(hashes[i][2], hashes[j][2])

                    if ssim_val >= self.ssim_threshold:
                        duplicates.append({
                            "image_a": hashes[i][1],
                            "image_b": hashes[j][1],
                            "ssim": round(ssim_val, 4),
                            "hash_distance": hash_dist
                        })

        # Compute unique images (remove duplicates)
        duplicate_files = set()
        for d in duplicates:
            duplicate_files.add(d["image_b"])  # Keep first, mark second

        unique_count = len(images) - len(duplicate_files)

        return {
            "total_images": len(images),
            "unique_images": unique_count,
            "duplicate_pairs": len(duplicates),
            "total_comparisons": total_comparisons,
            "duplicates": duplicates,
            "files_to_remove": sorted(list(duplicate_files))
        }
