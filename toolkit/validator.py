"""
Image Quality Validation Module
Evaluates images against configurable quality thresholds to filter
low-quality samples before they enter an AI/CV training dataset.

Quality Metrics:
    - Blur Score: Laplacian variance (higher = sharper)
    - Brightness: Mean luminance in grayscale
    - Contrast: Standard deviation of pixel intensities
    - Resolution: Minimum width/height dimensions
    - Aspect Ratio: Width-to-height ratio bounds
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class QualityThresholds:
    """
    Configurable thresholds for image quality validation.
    
    Attributes:
        blur_threshold: Minimum Laplacian variance to pass sharpness check.
                        Lower values = more blur tolerance. Default: 100.0
        min_brightness: Minimum mean luminance (0-255). Default: 40
        max_brightness: Maximum mean luminance (0-255). Default: 220
        min_contrast: Minimum pixel intensity std deviation. Default: 25.0
        min_resolution: Minimum dimension (width or height) in pixels. Default: 224
        min_aspect_ratio: Minimum width/height ratio. Default: 0.25
        max_aspect_ratio: Maximum width/height ratio. Default: 4.0
    """
    blur_threshold: float = 100.0
    min_brightness: int = 40
    max_brightness: int = 220
    min_contrast: float = 25.0
    min_resolution: int = 224
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 4.0


@dataclass
class QualityResult:
    """
    Stores the quality validation result for a single image.
    
    Attributes:
        filename: Name of the image file.
        passed: Whether the image passed all quality checks.
        blur_score: Laplacian variance score (higher = sharper).
        brightness: Mean luminance value (0-255).
        contrast: Pixel intensity standard deviation.
        resolution: (width, height) tuple.
        aspect_ratio: Width divided by height.
        checks: Dict of individual check results (True = passed).
        issues: List of human-readable issue descriptions.
    """
    filename: str = ""
    passed: bool = True
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    resolution: tuple = (0, 0)
    aspect_ratio: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for JSON/CSV export."""
        data = asdict(self)
        data["resolution"] = f"{self.resolution[0]}x{self.resolution[1]}"
        return data


class ImageValidator:
    """
    Validates images against quality thresholds for dataset QA.
    
    Uses OpenCV to compute quality metrics and flags images
    that fall below configurable thresholds.
    """

    def __init__(self, thresholds: QualityThresholds = None):
        """
        Initialize validator with quality thresholds.
        
        Args:
            thresholds: QualityThresholds instance. Uses defaults if None.
        """
        self.thresholds = thresholds or QualityThresholds()

    def compute_blur_score(self, image: np.ndarray) -> float:
        """
        Compute sharpness score using Laplacian variance.
        Higher values indicate sharper images.
        
        The Laplacian operator highlights regions of rapid intensity change.
        A well-focused image has strong edges, producing high variance.
        A blurry image has smooth transitions, producing low variance.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Laplacian variance as a float.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    def compute_brightness(self, image: np.ndarray) -> float:
        """
        Compute mean brightness (luminance) of the image.
        
        Converts to grayscale and computes the mean pixel value.
        Range: 0 (completely black) to 255 (completely white).
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Mean brightness as a float.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def compute_contrast(self, image: np.ndarray) -> float:
        """
        Compute contrast as the standard deviation of pixel intensities.
        
        Higher values indicate more contrast (wider range of intensities).
        Low values suggest a flat, washed-out image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Standard deviation of grayscale pixel values.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def get_resolution(self, image: np.ndarray) -> tuple:
        """
        Get image resolution as (width, height).
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Tuple of (width, height).
        """
        h, w = image.shape[:2]
        return (w, h)

    def get_aspect_ratio(self, image: np.ndarray) -> float:
        """
        Compute aspect ratio (width / height).
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Aspect ratio as a float.
        """
        h, w = image.shape[:2]
        return round(w / h, 3) if h > 0 else 0.0

    def validate(self, image: np.ndarray, filename: str = "") -> QualityResult:
        """
        Run all quality checks on a single image.
        
        Args:
            image: Input image as numpy array (BGR format).
            filename: Original filename for the report.
            
        Returns:
            QualityResult with all metrics, check results, and issues.
        """
        result = QualityResult(filename=filename)

        # --- Blur Check ---
        result.blur_score = round(self.compute_blur_score(image), 2)
        blur_passed = result.blur_score >= self.thresholds.blur_threshold
        result.checks["sharpness"] = blur_passed
        if not blur_passed:
            result.issues.append(
                f"Image is blurry (score: {result.blur_score}, "
                f"min: {self.thresholds.blur_threshold})"
            )

        # --- Brightness Check ---
        result.brightness = round(self.compute_brightness(image), 2)
        too_dark = result.brightness < self.thresholds.min_brightness
        too_bright = result.brightness > self.thresholds.max_brightness
        brightness_passed = not too_dark and not too_bright
        result.checks["brightness"] = brightness_passed
        if too_dark:
            result.issues.append(
                f"Image is too dark (brightness: {result.brightness}, "
                f"min: {self.thresholds.min_brightness})"
            )
        if too_bright:
            result.issues.append(
                f"Image is too bright (brightness: {result.brightness}, "
                f"max: {self.thresholds.max_brightness})"
            )

        # --- Contrast Check ---
        result.contrast = round(self.compute_contrast(image), 2)
        contrast_passed = result.contrast >= self.thresholds.min_contrast
        result.checks["contrast"] = contrast_passed
        if not contrast_passed:
            result.issues.append(
                f"Low contrast (score: {result.contrast}, "
                f"min: {self.thresholds.min_contrast})"
            )

        # --- Resolution Check ---
        result.resolution = self.get_resolution(image)
        min_dim = min(result.resolution)
        resolution_passed = min_dim >= self.thresholds.min_resolution
        result.checks["resolution"] = resolution_passed
        if not resolution_passed:
            result.issues.append(
                f"Resolution too low ({result.resolution[0]}x{result.resolution[1]}, "
                f"min dimension: {self.thresholds.min_resolution}px)"
            )

        # --- Aspect Ratio Check ---
        result.aspect_ratio = self.get_aspect_ratio(image)
        ar_passed = (self.thresholds.min_aspect_ratio
                     <= result.aspect_ratio
                     <= self.thresholds.max_aspect_ratio)
        result.checks["aspect_ratio"] = ar_passed
        if not ar_passed:
            result.issues.append(
                f"Extreme aspect ratio ({result.aspect_ratio}, "
                f"range: {self.thresholds.min_aspect_ratio}-{self.thresholds.max_aspect_ratio})"
            )

        # --- Overall Result ---
        result.passed = all(result.checks.values())

        return result

    def validate_batch(self, images: List[tuple]) -> List[QualityResult]:
        """
        Validate a batch of images.
        
        Args:
            images: List of (image_array, filename) tuples.
            
        Returns:
            List of QualityResult objects.
        """
        results = []
        for image, filename in images:
            result = self.validate(image, filename)
            results.append(result)
        return results

    def get_summary(self, results: List[QualityResult]) -> Dict[str, Any]:
        """
        Generate a summary of validation results for the entire dataset.
        
        Args:
            results: List of QualityResult objects.
            
        Returns:
            Dictionary with pass/fail counts and most common issues.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Count issue types
        issue_counts = {}
        for r in results:
            for issue in r.issues:
                # Extract issue type (e.g., "blurry", "too dark")
                issue_type = issue.split("(")[0].strip()
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            "total_images": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "issue_breakdown": issue_counts,
            "avg_blur_score": round(np.mean([r.blur_score for r in results]), 2) if results else 0,
            "avg_brightness": round(np.mean([r.brightness for r in results]), 2) if results else 0,
            "avg_contrast": round(np.mean([r.contrast for r in results]), 2) if results else 0,
        }
