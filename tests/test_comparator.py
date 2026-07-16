"""
Unit tests for the ImageComparator module.
Tests SSIM, histogram comparison, dHash, and duplicate detection.
"""

import pytest
import numpy as np
import cv2
from toolkit.comparator import ImageComparator


@pytest.fixture
def sample_image():
    """Create a test image with distinct features."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(img, (200, 100), 40, (255, 0, 0), -1)
    return img


@pytest.fixture
def different_image():
    """Create a visually different image."""
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (290, 190), (0, 0, 128), -1)
    return img


@pytest.fixture
def comparator():
    """Create comparator with default thresholds."""
    return ImageComparator()


class TestSSIM:
    """Test SSIM computation."""

    def test_identical_images_ssim_one(self, comparator, sample_image):
        ssim = comparator.compute_ssim(sample_image, sample_image.copy())
        assert ssim > 0.99

    def test_different_images_low_ssim(self, comparator, sample_image, different_image):
        ssim = comparator.compute_ssim(sample_image, different_image)
        assert ssim < 0.5

    def test_ssim_range(self, comparator, sample_image, different_image):
        ssim = comparator.compute_ssim(sample_image, different_image)
        assert -1 <= ssim <= 1

    def test_ssim_different_sizes(self, comparator, sample_image):
        small = cv2.resize(sample_image, (100, 75))
        ssim = comparator.compute_ssim(sample_image, small)
        assert isinstance(ssim, float)


class TestHistogramComparison:
    """Test histogram-based comparison."""

    def test_identical_histogram_correlation_one(self, comparator, sample_image):
        score = comparator.compute_histogram_similarity(sample_image, sample_image.copy())
        assert score > 0.99

    def test_different_histogram_lower(self, comparator, sample_image, different_image):
        score = comparator.compute_histogram_similarity(sample_image, different_image)
        assert score < 0.9


class TestDHash:
    """Test perceptual hashing."""

    def test_identical_images_zero_distance(self, comparator, sample_image):
        hash_a = comparator.compute_dhash(sample_image)
        hash_b = comparator.compute_dhash(sample_image.copy())
        assert comparator.compute_hash_distance(hash_a, hash_b) == 0

    def test_different_images_nonzero_distance(self, comparator, sample_image, different_image):
        hash_a = comparator.compute_dhash(sample_image)
        hash_b = comparator.compute_dhash(different_image)
        assert comparator.compute_hash_distance(hash_a, hash_b) > 0

    def test_hash_is_integer(self, comparator, sample_image):
        h = comparator.compute_dhash(sample_image)
        assert isinstance(h, int)

    def test_slightly_modified_low_distance(self, comparator, sample_image):
        modified = sample_image.copy()
        modified[0:5, 0:5] = 255  # tiny modification
        hash_a = comparator.compute_dhash(sample_image)
        hash_b = comparator.compute_dhash(modified)
        dist = comparator.compute_hash_distance(hash_a, hash_b)
        assert dist < 10


class TestCompare:
    """Test full comparison."""

    def test_compare_returns_result(self, comparator, sample_image, different_image):
        result = comparator.compare(sample_image, different_image, "a.jpg", "b.jpg")
        assert result.image_a == "a.jpg"
        assert result.image_b == "b.jpg"
        assert isinstance(result.ssim, float)
        assert isinstance(result.hash_distance, int)

    def test_identical_marked_duplicate(self, comparator, sample_image):
        result = comparator.compare(sample_image, sample_image.copy())
        assert result.is_duplicate is True

    def test_different_not_duplicate(self, comparator, sample_image, different_image):
        result = comparator.compare(sample_image, different_image)
        assert result.is_duplicate is False


class TestDuplicateFinder:
    """Test batch duplicate detection."""

    def test_finds_exact_duplicates(self, comparator, sample_image):
        images = [
            (sample_image, "img1.jpg"),
            (sample_image.copy(), "img2.jpg"),
            (sample_image.copy(), "img3.jpg"),
        ]
        report = comparator.find_duplicates(images)
        assert report["duplicate_pairs"] > 0
        assert report["unique_images"] < 3

    def test_no_duplicates_in_unique_set(self, comparator, sample_image, different_image):
        blank = np.full((200, 300, 3), 128, dtype=np.uint8)
        images = [
            (sample_image, "img1.jpg"),
            (different_image, "img2.jpg"),
            (blank, "img3.jpg"),
        ]
        report = comparator.find_duplicates(images)
        assert report["unique_images"] == 3

    def test_report_structure(self, comparator, sample_image):
        images = [(sample_image, "test.jpg")]
        report = comparator.find_duplicates(images)
        assert "total_images" in report
        assert "unique_images" in report
        assert "duplicate_pairs" in report
        assert "duplicates" in report
        assert "files_to_remove" in report

    def test_to_dict(self, comparator, sample_image, different_image):
        result = comparator.compare(sample_image, different_image)
        d = result.to_dict()
        assert "ssim" in d
        assert "hash_distance" in d
        assert "is_duplicate" in d
