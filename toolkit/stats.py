"""
Dataset Statistics Module
Computes comprehensive statistics about an image dataset for analysis.

Metrics:
    - Resolution distribution (min, max, mean, median)
    - File size distribution
    - Color channel histograms (B, G, R distributions)
    - Format breakdown
    - Quality score distribution
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class DatasetStats:
    """
    Stores computed statistics for an image dataset.

    Attributes:
        total_images: Number of images in the dataset.
        total_size_bytes: Total file size in bytes.
        resolution: Min/max/mean/median resolution statistics.
        file_sizes: Min/max/mean/median file size statistics.
        formats: Count of each file format.
        color_stats: Per-channel mean and std for B, G, R.
        quality_distribution: Histogram of blur scores.
    """
    total_images: int = 0
    total_size_bytes: int = 0
    resolution: Dict[str, Any] = field(default_factory=dict)
    file_sizes: Dict[str, Any] = field(default_factory=dict)
    formats: Dict[str, int] = field(default_factory=dict)
    color_stats: Dict[str, Any] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "total_images": self.total_images,
            "total_size_mb": round(self.total_size_bytes / (1024 * 1024), 2),
            "resolution": self.resolution,
            "file_sizes": self.file_sizes,
            "formats": self.formats,
            "color_stats": self.color_stats,
            "quality_distribution": self.quality_distribution
        }


class DatasetAnalyzer:
    """
    Analyzes an image dataset and computes comprehensive statistics.
    """

    def analyze(self, image_paths: List[Path]) -> DatasetStats:
        """
        Compute statistics for a dataset of images.

        Args:
            image_paths: List of Path objects pointing to image files.

        Returns:
            DatasetStats with all computed metrics.
        """
        stats = DatasetStats()
        stats.total_images = len(image_paths)

        widths = []
        heights = []
        file_sizes = []
        channel_means = {"B": [], "G": [], "R": []}
        channel_stds = {"B": [], "G": [], "R": []}
        blur_scores = []
        formats = {}

        for path in image_paths:
            # File metadata
            size = path.stat().st_size
            file_sizes.append(size)
            stats.total_size_bytes += size

            ext = path.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1

            # Load image
            image = cv2.imread(str(path))
            if image is None:
                continue

            h, w = image.shape[:2]
            widths.append(w)
            heights.append(h)

            # Color statistics per channel
            for i, ch in enumerate(["B", "G", "R"]):
                channel_means[ch].append(float(np.mean(image[:, :, i])))
                channel_stds[ch].append(float(np.std(image[:, :, i])))

            # Blur score (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
            blur_scores.append(blur)

        # Resolution stats
        if widths:
            stats.resolution = {
                "width": {
                    "min": int(min(widths)),
                    "max": int(max(widths)),
                    "mean": round(float(np.mean(widths)), 1),
                    "median": int(np.median(widths))
                },
                "height": {
                    "min": int(min(heights)),
                    "max": int(max(heights)),
                    "mean": round(float(np.mean(heights)), 1),
                    "median": int(np.median(heights))
                },
                "total_pixels_avg": int(np.mean(widths) * np.mean(heights))
            }

        # File size stats
        if file_sizes:
            stats.file_sizes = {
                "min_kb": round(min(file_sizes) / 1024, 1),
                "max_kb": round(max(file_sizes) / 1024, 1),
                "mean_kb": round(float(np.mean(file_sizes)) / 1024, 1),
                "median_kb": round(float(np.median(file_sizes)) / 1024, 1)
            }

        # Format breakdown
        stats.formats = formats

        # Color stats
        if channel_means["B"]:
            stats.color_stats = {}
            for ch in ["B", "G", "R"]:
                stats.color_stats[ch] = {
                    "mean_intensity": round(float(np.mean(channel_means[ch])), 1),
                    "std_intensity": round(float(np.mean(channel_stds[ch])), 1)
                }

        # Quality distribution (bucket blur scores)
        if blur_scores:
            stats.quality_distribution = self._bucket_scores(blur_scores)

        return stats

    def _bucket_scores(self, scores: List[float]) -> Dict[str, int]:
        """
        Bucket blur scores into quality categories.

        Categories:
            - very_blurry: score < 50
            - blurry: 50 <= score < 100
            - acceptable: 100 <= score < 500
            - sharp: 500 <= score < 2000
            - very_sharp: score >= 2000

        Args:
            scores: List of blur scores.

        Returns:
            Dict mapping category name to count.
        """
        buckets = {
            "very_blurry (<50)": 0,
            "blurry (50-100)": 0,
            "acceptable (100-500)": 0,
            "sharp (500-2000)": 0,
            "very_sharp (2000+)": 0
        }

        for s in scores:
            if s < 50:
                buckets["very_blurry (<50)"] += 1
            elif s < 100:
                buckets["blurry (50-100)"] += 1
            elif s < 500:
                buckets["acceptable (100-500)"] += 1
            elif s < 2000:
                buckets["sharp (500-2000)"] += 1
            else:
                buckets["very_sharp (2000+)"] += 1

        return buckets

    def compute_channel_histograms(self, image_paths: List[Path],
                                    bins: int = 64) -> Dict[str, List[int]]:
        """
        Compute aggregated color histograms across the entire dataset.

        Args:
            image_paths: List of image file paths.
            bins: Number of histogram bins (default: 64).

        Returns:
            Dict with B, G, R keys mapping to histogram bin counts.
        """
        total_hist = {"B": np.zeros(bins), "G": np.zeros(bins), "R": np.zeros(bins)}

        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                continue

            for i, ch in enumerate(["B", "G", "R"]):
                hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                total_hist[ch] += hist.flatten()

        return {
            ch: [int(v) for v in total_hist[ch]]
            for ch in ["B", "G", "R"]
        }
