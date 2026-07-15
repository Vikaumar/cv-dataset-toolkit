"""
Export Module
Handles saving augmented images, augmentation metadata logs,
and quality validation reports in JSON and CSV formats.
"""

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple


class DatasetExporter:
    """
    Exports augmented images and validation reports to structured directories.
    
    Output structure:
        output_dir/
        ├── augmented/       # Augmented image files
        ├── metadata/        # Augmentation logs (JSON)
        └── reports/         # Quality validation reports (JSON + CSV)
    """

    def __init__(self, output_dir: str):
        """
        Initialize exporter and create output directory structure.
        
        Args:
            output_dir: Root output directory path.
        """
        self.output_dir = Path(output_dir)
        self.augmented_dir = self.output_dir / "augmented"
        self.metadata_dir = self.output_dir / "metadata"
        self.reports_dir = self.output_dir / "reports"

        # Create directories
        for d in [self.augmented_dir, self.metadata_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_augmented_image(self, image: np.ndarray, original_name: str,
                              metadata: Dict[str, Any]) -> str:
        """
        Save an augmented image with a descriptive filename.
        
        Filename format: {original_stem}_{transform}_{param}.{ext}
        Example: photo1_rotate_90.jpg, photo1_gaussian_noise_sigma30.jpg
        
        Args:
            image: Augmented image as numpy array.
            original_name: Original image filename.
            metadata: Augmentation metadata dict.
            
        Returns:
            Path to saved image as string.
        """
        stem = Path(original_name).stem
        ext = Path(original_name).suffix or ".jpg"
        transform = metadata.get("transform", "unknown")

        # Build descriptive suffix from metadata
        params = {k: v for k, v in metadata.items() if k != "transform"}
        if params:
            # Flatten param values into filename-safe string
            param_parts = []
            for k, v in params.items():
                if isinstance(v, dict):
                    continue  # Skip nested dicts (like crop region)
                param_parts.append(f"{k}{v}")
            param_str = "_".join(param_parts) if param_parts else ""
            filename = f"{stem}_{transform}_{param_str}{ext}" if param_str else f"{stem}_{transform}{ext}"
        else:
            filename = f"{stem}_{transform}{ext}"

        # Sanitize filename
        filename = filename.replace(" ", "_").replace(".", "_", filename.count(".") - 1)

        output_path = self.augmented_dir / filename
        cv2.imwrite(str(output_path), image)
        return str(output_path)

    def save_augmented_batch(self, results: List[Tuple[np.ndarray, Dict[str, Any]]],
                              original_name: str) -> List[str]:
        """
        Save a batch of augmented images from a single source image.
        
        Args:
            results: List of (augmented_image, metadata) tuples.
            original_name: Original image filename.
            
        Returns:
            List of saved file paths.
        """
        saved_paths = []
        for i, (image, metadata) in enumerate(results):
            # Add index to avoid filename collisions
            metadata_with_idx = {**metadata, "variant_index": i}
            path = self.save_augmented_image(image, original_name, metadata)
            saved_paths.append(path)
        return saved_paths

    def save_augmentation_log(self, log_entries: List[Dict[str, Any]]) -> str:
        """
        Save augmentation metadata log as a JSON file.
        
        The log records every augmentation applied, enabling
        full reproducibility of the augmented dataset.
        
        Args:
            log_entries: List of metadata dicts from augmentation operations.
            
        Returns:
            Path to saved JSON log.
        """
        log = {
            "generated_at": datetime.now().isoformat(),
            "total_augmentations": len(log_entries),
            "entries": log_entries
        }

        log_path = self.metadata_dir / "augmentation_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

        return str(log_path)

    def save_quality_report_json(self, results: List[Any],
                                  summary: Dict[str, Any]) -> str:
        """
        Save quality validation results as a JSON report.
        
        Args:
            results: List of QualityResult objects.
            summary: Summary dict from ImageValidator.get_summary().
            
        Returns:
            Path to saved JSON report.
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "results": [r.to_dict() for r in results]
        }

        report_path = self.reports_dir / "quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return str(report_path)

    def save_quality_report_csv(self, results: List[Any]) -> str:
        """
        Save quality validation results as a CSV report.
        
        Each row represents one image with its quality metrics
        and pass/fail status for each check.
        
        Args:
            results: List of QualityResult objects.
            
        Returns:
            Path to saved CSV report.
        """
        report_path = self.reports_dir / "quality_report.csv"

        if not results:
            return str(report_path)

        fieldnames = [
            "filename", "passed", "blur_score", "brightness",
            "contrast", "resolution", "aspect_ratio",
            "sharpness_ok", "brightness_ok", "contrast_ok",
            "resolution_ok", "aspect_ratio_ok", "issues"
        ]

        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                row = {
                    "filename": r.filename,
                    "passed": r.passed,
                    "blur_score": r.blur_score,
                    "brightness": r.brightness,
                    "contrast": r.contrast,
                    "resolution": f"{r.resolution[0]}x{r.resolution[1]}",
                    "aspect_ratio": r.aspect_ratio,
                    "sharpness_ok": r.checks.get("sharpness", ""),
                    "brightness_ok": r.checks.get("brightness", ""),
                    "contrast_ok": r.checks.get("contrast", ""),
                    "resolution_ok": r.checks.get("resolution", ""),
                    "aspect_ratio_ok": r.checks.get("aspect_ratio", ""),
                    "issues": "; ".join(r.issues) if r.issues else "None"
                }
                writer.writerow(row)

        return str(report_path)
