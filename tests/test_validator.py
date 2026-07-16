"""
Unit tests for the ImageValidator module.
Tests all 8 quality checks with known test images.
"""

import pytest
import numpy as np
import cv2
from toolkit.validator import ImageValidator, QualityThresholds, QualityResult


@pytest.fixture
def sharp_image():
    """Create a sharp image with clear edges."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 255), 3)
    cv2.circle(img, (400, 240), 100, (255, 100, 50), -1)
    cv2.putText(img, "SHARP", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img


@pytest.fixture
def blurry_image(sharp_image):
    """Create a blurry version of the sharp image."""
    return cv2.GaussianBlur(sharp_image, (31, 31), 0)


@pytest.fixture
def dark_image(sharp_image):
    """Create a very dark image."""
    return (sharp_image * 0.1).astype(np.uint8)


@pytest.fixture
def bright_image():
    """Create a very bright image."""
    return np.full((480, 640, 3), 250, dtype=np.uint8)


@pytest.fixture
def low_res_image():
    """Create a low resolution image."""
    return np.zeros((50, 50, 3), dtype=np.uint8)


@pytest.fixture
def validator():
    """Create validator with default thresholds."""
    return ImageValidator()


class TestBlurDetection:
    """Test blur/sharpness detection."""

    def test_sharp_image_passes(self, validator, sharp_image):
        result = validator.validate(sharp_image)
        assert result.checks["sharpness"] is True
        assert result.blur_score > 100

    def test_blurry_image_fails(self, validator, blurry_image):
        result = validator.validate(blurry_image)
        assert result.checks["sharpness"] is False
        assert result.blur_score < 100

    def test_blur_score_positive(self, validator, sharp_image):
        score = validator.compute_blur_score(sharp_image)
        assert score > 0


class TestBrightnessDetection:
    """Test brightness detection."""

    def test_dark_image_fails(self, validator, dark_image):
        result = validator.validate(dark_image)
        assert result.checks["brightness"] is False
        assert result.brightness < 40

    def test_bright_image_fails(self, validator, bright_image):
        result = validator.validate(bright_image)
        assert result.checks["brightness"] is False

    def test_normal_brightness_passes(self, validator, sharp_image):
        # Adjust sharp_image to have normal brightness
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = validator.validate(img)
        assert result.checks["brightness"] is True


class TestContrastDetection:
    """Test contrast detection."""

    def test_flat_image_fails(self, validator):
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = validator.validate(flat)
        assert result.checks["contrast"] is False

    def test_high_contrast_passes(self, validator, sharp_image):
        result = validator.validate(sharp_image)
        assert result.contrast > 0


class TestResolutionCheck:
    """Test resolution validation."""

    def test_low_res_fails(self, validator, low_res_image):
        result = validator.validate(low_res_image)
        assert result.checks["resolution"] is False

    def test_normal_res_passes(self, validator, sharp_image):
        result = validator.validate(sharp_image)
        assert result.checks["resolution"] is True

    def test_resolution_tuple(self, validator, sharp_image):
        res = validator.get_resolution(sharp_image)
        assert res == (640, 480)


class TestAspectRatio:
    """Test aspect ratio validation."""

    def test_extreme_aspect_ratio_fails(self, validator):
        extreme = np.zeros((10, 500, 3), dtype=np.uint8)
        thresholds = QualityThresholds(max_aspect_ratio=4.0)
        val = ImageValidator(thresholds)
        result = val.validate(extreme)
        assert result.checks["aspect_ratio"] is False

    def test_normal_aspect_ratio_passes(self, validator, sharp_image):
        result = validator.validate(sharp_image)
        assert result.checks["aspect_ratio"] is True


class TestNoiseDetection:
    """Test noise level detection."""

    def test_clean_image_passes(self, validator, sharp_image):
        result = validator.validate(sharp_image)
        assert result.checks["noise"] is True

    def test_noisy_image_detected(self, validator, sharp_image):
        noisy = sharp_image.copy()
        noise = np.random.normal(0, 50, noisy.shape).astype(np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = validator.validate(noisy)
        assert result.noise_level > 5


class TestSaturationDetection:
    """Test saturation detection."""

    def test_grayscale_detected(self, validator):
        gray = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = validator.validate(gray)
        assert result.saturation < 10

    def test_colorful_image_passes(self, validator):
        colorful = np.zeros((200, 200, 3), dtype=np.uint8)
        colorful[:, :100] = [255, 0, 0]
        colorful[:, 100:] = [0, 255, 0]
        result = validator.validate(colorful)
        assert result.saturation > 10


class TestBatchValidation:
    """Test batch processing."""

    def test_batch_returns_correct_count(self, validator, sharp_image, blurry_image):
        results = validator.validate_batch([
            (sharp_image, "sharp.jpg"),
            (blurry_image, "blurry.jpg"),
        ])
        assert len(results) == 2

    def test_summary_statistics(self, validator, sharp_image, blurry_image):
        results = validator.validate_batch([
            (sharp_image, "sharp.jpg"),
            (blurry_image, "blurry.jpg"),
        ])
        summary = validator.get_summary(results)
        assert summary["total_images"] == 2
        assert summary["passed"] + summary["failed"] == 2


class TestQualityResult:
    """Test QualityResult data class."""

    def test_to_dict(self):
        result = QualityResult(
            filename="test.jpg", passed=True,
            blur_score=500.0, resolution=(640, 480)
        )
        d = result.to_dict()
        assert d["resolution"] == "640x480"
        assert d["filename"] == "test.jpg"

    def test_overall_pass_requires_all(self, validator):
        # An image that fails at least one check should have passed=False
        dark = (np.zeros((300, 300, 3), dtype=np.uint8) + 10)
        result = validator.validate(dark)
        assert result.passed is False


class TestCustomThresholds:
    """Test configurable thresholds."""

    def test_relaxed_blur_threshold(self, blurry_image):
        relaxed = QualityThresholds(blur_threshold=0.5)
        val = ImageValidator(relaxed)
        result = val.validate(blurry_image)
        assert result.checks["sharpness"] is True

    def test_strict_resolution(self, sharp_image):
        strict = QualityThresholds(min_resolution=1000)
        val = ImageValidator(strict)
        result = val.validate(sharp_image)
        assert result.checks["resolution"] is False
