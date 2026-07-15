"""
CV Dataset Toolkit — CLI Entry Point

Usage:
    python main.py augment  --input <dir> --output <dir> [--count N] [--transforms ...]
    python main.py validate --input <dir> --output <dir> [--min-resolution N] [--blur-threshold N]
    python main.py pipeline --input <dir> --output <dir> [--count N]

Commands:
    augment   — Apply random augmentations to all images in the input directory.
    validate  — Run quality checks on all images and generate reports.
    pipeline  — Run augmentation followed by validation on the augmented output.
"""

import argparse
import sys
import cv2
from pathlib import Path
from typing import List

from toolkit.augmentor import ImageAugmentor
from toolkit.validator import ImageValidator, QualityThresholds
from toolkit.exporter import DatasetExporter


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_dir: str) -> List[Path]:
    """
    Recursively find all image files in a directory.
    
    Args:
        input_dir: Path to directory to scan.
        
    Returns:
        Sorted list of image file paths.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    images = [
        f for f in sorted(input_path.rglob("*"))
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    ]

    if not images:
        print(f"[WARNING] No images found in: {input_dir}")
        print(f"  Supported formats: {', '.join(IMAGE_EXTENSIONS)}")

    return images


def load_image(path: Path):
    """
    Load an image from disk using OpenCV.
    
    Args:
        path: Path to image file.
        
    Returns:
        Image as numpy array (BGR), or None if loading fails.
    """
    image = cv2.imread(str(path))
    if image is None:
        print(f"  [SKIP] Could not read: {path.name}")
    return image


def cmd_augment(args):
    """Execute the 'augment' command."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Image Augmentation")
    print("=" * 60)

    images = find_images(args.input)
    if not images:
        return

    print(f"\n[INFO] Found {len(images)} images in: {args.input}")
    print(f"[INFO] Generating {args.count} augmented variant(s) per image")
    if args.transforms:
        print(f"[INFO] Using transforms: {', '.join(args.transforms)}")
    print()

    augmentor = ImageAugmentor(seed=args.seed)
    exporter = DatasetExporter(args.output)

    all_log_entries = []
    total_saved = 0

    for img_path in images:
        image = load_image(img_path)
        if image is None:
            continue

        print(f"  Processing: {img_path.name}")

        # Apply augmentations
        results = augmentor.apply_random(
            image,
            transforms=args.transforms,
            count=args.count
        )

        # Save augmented images
        saved_paths = exporter.save_augmented_batch(results, img_path.name)
        total_saved += len(saved_paths)

        # Log metadata
        for (_, metadata), saved_path in zip(results, saved_paths):
            log_entry = {
                "source_image": img_path.name,
                "output_path": saved_path,
                **metadata
            }
            all_log_entries.append(log_entry)

        print(f"    -> Generated {len(results)} variants")

    # Save augmentation log
    log_path = exporter.save_augmentation_log(all_log_entries)

    print(f"\n{'=' * 60}")
    print(f"  AUGMENTATION COMPLETE")
    print(f"  Total images saved:  {total_saved}")
    print(f"  Output directory:    {args.output}")
    print(f"  Augmentation log:    {log_path}")
    print(f"{'=' * 60}")


def cmd_validate(args):
    """Execute the 'validate' command."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Image Quality Validation")
    print("=" * 60)

    images = find_images(args.input)
    if not images:
        return

    print(f"\n[INFO] Found {len(images)} images in: {args.input}")
    print(f"[INFO] Blur threshold: {args.blur_threshold}")
    print(f"[INFO] Min resolution: {args.min_resolution}px")
    print()

    # Configure thresholds
    thresholds = QualityThresholds(
        blur_threshold=args.blur_threshold,
        min_resolution=args.min_resolution,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
    )
    validator = ImageValidator(thresholds)
    exporter = DatasetExporter(args.output)

    results = []

    for img_path in images:
        image = load_image(img_path)
        if image is None:
            continue

        result = validator.validate(image, img_path.name)
        results.append(result)

        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status}  {img_path.name}")
        print(f"         Blur: {result.blur_score:>8.1f}  |  "
              f"Brightness: {result.brightness:>5.1f}  |  "
              f"Contrast: {result.contrast:>5.1f}  |  "
              f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
        if result.issues:
            for issue in result.issues:
                print(f"         [!] {issue}")
        print()

    # Generate summary and save reports
    summary = validator.get_summary(results)
    json_path = exporter.save_quality_report_json(results, summary)
    csv_path = exporter.save_quality_report_csv(results)

    print(f"{'=' * 60}")
    print(f"  VALIDATION SUMMARY")
    print(f"  Total images:   {summary['total_images']}")
    print(f"  Passed:         {summary['passed']} ({summary['pass_rate']}%)")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Avg blur score: {summary['avg_blur_score']}")
    print(f"  Avg brightness: {summary['avg_brightness']}")
    print(f"  Avg contrast:   {summary['avg_contrast']}")
    if summary["issue_breakdown"]:
        print(f"\n  Issue Breakdown:")
        for issue_type, count in summary["issue_breakdown"].items():
            print(f"    - {issue_type}: {count}")
    print(f"\n  JSON report: {json_path}")
    print(f"  CSV report:  {csv_path}")
    print(f"{'=' * 60}")


def cmd_pipeline(args):
    """Execute the 'pipeline' command (augment + validate)."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Full Pipeline (Augment + Validate)")
    print("=" * 60)
    print()

    # Step 1: Augment
    args.output_augment = str(Path(args.output))
    augment_args = argparse.Namespace(
        input=args.input,
        output=args.output,
        count=args.count,
        transforms=args.transforms,
        seed=args.seed,
    )
    cmd_augment(augment_args)

    # Step 2: Validate the augmented output
    print(f"\n{'~' * 60}")
    print(f"  Now validating augmented images...")
    print(f"{'~' * 60}\n")

    augmented_dir = str(Path(args.output) / "augmented")
    validate_args = argparse.Namespace(
        input=augmented_dir,
        output=args.output,
        blur_threshold=args.blur_threshold,
        min_resolution=args.min_resolution,
        min_brightness=40,
        max_brightness=220,
    )
    cmd_validate(validate_args)


def main():
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="CV Dataset Toolkit",
        description="Image augmentation and quality validation for AI/CV datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py augment  --input ./images --output ./output --count 5\n"
               "  python main.py validate --input ./images --output ./output\n"
               "  python main.py pipeline --input ./images --output ./output --count 3\n"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Augment command ---
    aug_parser = subparsers.add_parser("augment", help="Augment images for dataset expansion")
    aug_parser.add_argument("--input", "-i", required=True,
                            help="Input directory containing images")
    aug_parser.add_argument("--output", "-o", required=True,
                            help="Output directory for augmented images")
    aug_parser.add_argument("--count", "-c", type=int, default=5,
                            help="Number of augmented variants per image (default: 5)")
    aug_parser.add_argument("--transforms", "-t", nargs="+", default=None,
                            choices=ImageAugmentor.AVAILABLE_TRANSFORMS,
                            help="Specific transforms to apply (default: all)")
    aug_parser.add_argument("--seed", "-s", type=int, default=None,
                            help="Random seed for reproducibility")

    # --- Validate command ---
    val_parser = subparsers.add_parser("validate", help="Validate image quality")
    val_parser.add_argument("--input", "-i", required=True,
                            help="Input directory containing images to validate")
    val_parser.add_argument("--output", "-o", required=True,
                            help="Output directory for quality reports")
    val_parser.add_argument("--blur-threshold", type=float, default=100.0,
                            help="Minimum blur score to pass (default: 100.0)")
    val_parser.add_argument("--min-resolution", type=int, default=224,
                            help="Minimum image dimension in pixels (default: 224)")
    val_parser.add_argument("--min-brightness", type=int, default=40,
                            help="Minimum brightness value 0-255 (default: 40)")
    val_parser.add_argument("--max-brightness", type=int, default=220,
                            help="Maximum brightness value 0-255 (default: 220)")

    # --- Pipeline command ---
    pipe_parser = subparsers.add_parser("pipeline",
                                         help="Run augmentation then validation")
    pipe_parser.add_argument("--input", "-i", required=True,
                              help="Input directory containing images")
    pipe_parser.add_argument("--output", "-o", required=True,
                              help="Output directory for all results")
    pipe_parser.add_argument("--count", "-c", type=int, default=5,
                              help="Augmented variants per image (default: 5)")
    pipe_parser.add_argument("--transforms", "-t", nargs="+", default=None,
                              choices=ImageAugmentor.AVAILABLE_TRANSFORMS,
                              help="Specific transforms to apply")
    pipe_parser.add_argument("--seed", "-s", type=int, default=None,
                              help="Random seed for reproducibility")
    pipe_parser.add_argument("--blur-threshold", type=float, default=100.0,
                              help="Minimum blur score (default: 100.0)")
    pipe_parser.add_argument("--min-resolution", type=int, default=224,
                              help="Minimum dimension in pixels (default: 224)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "augment": cmd_augment,
        "validate": cmd_validate,
        "pipeline": cmd_pipeline,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
