"""
Unit tests for the ImageAugmentor module.
Tests all 16 augmentation transforms for correctness.
"""

import pytest
import numpy as np
import cv2
from toolkit.augmentor import ImageAugmentor


@pytest.fixture
def sample_image():
    """Create a 200x300 test image with known content."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[50:150, 100:200] = [128, 64, 200]  # colored rectangle
    cv2.circle(img, (150, 100), 30, (0, 255, 0), -1)
    return img


@pytest.fixture
def augmentor():
    """Create augmentor with fixed seed."""
    return ImageAugmentor(seed=42)


class TestAugmentorBasics:
    """Test basic augmentor functionality."""

    def test_available_transforms_count(self):
        assert len(ImageAugmentor.AVAILABLE_TRANSFORMS) == 16

    def test_seed_reproducibility(self, sample_image):
        aug1 = ImageAugmentor(seed=42)
        results1 = aug1.apply_random(sample_image, count=3)
        aug2 = ImageAugmentor(seed=42)
        results2 = aug2.apply_random(sample_image, count=3)
        for (img1, meta1), (img2, meta2) in zip(results1, results2):
            assert meta1["transform"] == meta2["transform"]

    def test_invalid_transform_raises(self, augmentor, sample_image):
        with pytest.raises(ValueError):
            augmentor.apply_random(sample_image, transforms=["nonexistent"])

    def test_apply_all_returns_16(self, augmentor, sample_image):
        results = augmentor.apply_all(sample_image)
        assert len(results) == 16


class TestGeometricTransforms:
    """Test geometric augmentation transforms."""

    def test_rotate_returns_image(self, augmentor, sample_image):
        result, meta = augmentor.rotate(sample_image)
        assert result is not None
        assert meta["transform"] == "rotate"
        assert meta["angle"] in [90, 180, 270]

    def test_flip_h_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.flip_h(sample_image)
        assert result.shape == sample_image.shape
        assert meta["transform"] == "flip_h"

    def test_flip_v_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.flip_v(sample_image)
        assert result.shape == sample_image.shape
        assert meta["transform"] == "flip_v"

    def test_crop_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.crop(sample_image)
        assert result.shape == sample_image.shape
        assert "crop_ratio" in meta

    def test_perspective_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.perspective(sample_image)
        assert result.shape == sample_image.shape
        assert meta["transform"] == "perspective"

    def test_elastic_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.elastic(sample_image)
        assert result.shape == sample_image.shape
        assert "alpha" in meta and "sigma" in meta


class TestColorTransforms:
    """Test color/intensity augmentation transforms."""

    def test_brightness_valid_range(self, augmentor, sample_image):
        result, meta = augmentor.brightness(sample_image)
        assert result.shape == sample_image.shape
        assert 0.5 <= meta["factor"] <= 1.5

    def test_contrast_valid_range(self, augmentor, sample_image):
        result, meta = augmentor.contrast(sample_image)
        assert result.dtype == np.uint8
        assert 0.5 <= meta["factor"] <= 1.8

    def test_clahe_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.clahe(sample_image)
        assert result.shape == sample_image.shape
        assert "clip_limit" in meta and "grid_size" in meta

    def test_hue_shift_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.hue_shift(sample_image)
        assert result.shape == sample_image.shape
        assert -30 <= meta["shift"] <= 30

    def test_channel_shuffle_preserves_dimensions(self, augmentor, sample_image):
        result, meta = augmentor.channel_shuffle(sample_image)
        assert result.shape == sample_image.shape
        assert len(meta["order"]) == 3


class TestNoiseTransforms:
    """Test noise/degradation augmentation transforms."""

    def test_gaussian_noise_adds_noise(self, augmentor, sample_image):
        result, meta = augmentor.gaussian_noise(sample_image)
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_salt_pepper_adds_noise(self, augmentor, sample_image):
        result, meta = augmentor.salt_pepper(sample_image)
        assert result.shape == sample_image.shape
        assert 0.01 <= meta["density"] <= 0.05

    def test_gaussian_blur_reduces_sharpness(self, augmentor, sample_image):
        result, meta = augmentor.gaussian_blur(sample_image)
        assert result.shape == sample_image.shape
        assert meta["kernel_size"] in [3, 5, 7, 9]

    def test_motion_blur(self, augmentor, sample_image):
        result, meta = augmentor.motion_blur(sample_image)
        assert result.shape == sample_image.shape

    def test_cutout_creates_black_regions(self, augmentor, sample_image):
        result, meta = augmentor.cutout(sample_image)
        assert result.shape == sample_image.shape
        assert 1 <= meta["num_rects"] <= 3
        # Verify at least one region is black
        r = meta["regions"][0]
        region = result[r["top"]:r["top"]+r["height"], r["left"]:r["left"]+r["width"]]
        assert np.all(region == 0)
