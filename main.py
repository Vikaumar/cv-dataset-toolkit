"""
CV Dataset Toolkit -- CLI Entry Point

Usage:
    python main.py augment  --input <dir> --output <dir> [--count N] [--transforms ...]
    python main.py validate --input <dir> --output <dir> [--min-resolution N] [--blur-threshold N]
    python main.py compare  --input <dir> --output <dir> [--ssim-threshold F] [--hash-threshold N]
    python main.py stats    --input <dir> --output <dir>
    python main.py pipeline --input <dir> --output <dir> [--count N]
    python main.py report   --input <dir> --output <dir>

Commands:
    augment  -- Apply random augmentations to all images in the input directory.
    validate -- Run quality checks on all images and generate reports.
    compare  -- Find duplicate/near-duplicate images in a directory.
    stats    -- Compute and display dataset statistics.
    pipeline -- Run augmentation + validation + stats + duplicates + HTML report.
    report   -- Generate HTML report from existing pipeline output.
"""

import argparse
import json
import sys
import cv2
from pathlib import Path
from typing import List

from toolkit.augmentor import ImageAugmentor
from toolkit.validator import ImageValidator, QualityThresholds
from toolkit.comparator import ImageComparator
from toolkit.stats import DatasetAnalyzer
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


# ============================================================
# COMMAND: augment
# ============================================================

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
    print(f"[INFO] Available transforms: {len(ImageAugmentor.AVAILABLE_TRANSFORMS)}")
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

    return all_log_entries


# ============================================================
# COMMAND: validate
# ============================================================

def cmd_validate(args):
    """Execute the 'validate' command."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Image Quality Validation")
    print("=" * 60)

    images = find_images(args.input)
    if not images:
        return None, None

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
              f"Noise: {result.noise_level:>5.1f}  |  "
              f"Sat: {result.saturation:>5.1f}")
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

    return results, summary


# ============================================================
# COMMAND: compare
# ============================================================

def cmd_compare(args):
    """Execute the 'compare' command."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Duplicate Detection")
    print("=" * 60)

    images = find_images(args.input)
    if not images:
        return None

    print(f"\n[INFO] Found {len(images)} images in: {args.input}")
    print(f"[INFO] SSIM threshold: {args.ssim_threshold}")
    print(f"[INFO] Hash threshold: {args.hash_threshold}")
    n_comparisons = len(images) * (len(images) - 1) // 2
    print(f"[INFO] Total comparisons: {n_comparisons}")
    print()

    comparator = ImageComparator(
        ssim_threshold=args.ssim_threshold,
        hash_threshold=args.hash_threshold
    )

    # Load all images
    loaded = []
    for img_path in images:
        image = load_image(img_path)
        if image is not None:
            loaded.append((image, img_path.name))

    print(f"  Scanning for duplicates...")
    report = comparator.find_duplicates(loaded)

    print(f"\n{'=' * 60}")
    print(f"  DUPLICATE DETECTION RESULTS")
    print(f"  Total images:     {report['total_images']}")
    print(f"  Unique images:    {report['unique_images']}")
    print(f"  Duplicate pairs:  {report['duplicate_pairs']}")
    print(f"  Comparisons made: {report['total_comparisons']}")

    if report["duplicates"]:
        print(f"\n  Duplicate Pairs:")
        for d in report["duplicates"]:
            print(f"    {d['image_a']} <-> {d['image_b']}  "
                  f"(SSIM: {d['ssim']}, hash dist: {d['hash_distance']})")

    if report["files_to_remove"]:
        print(f"\n  Suggested removals ({len(report['files_to_remove'])}):")
        for f in report["files_to_remove"]:
            print(f"    - {f}")

    print(f"{'=' * 60}")

    # Save report
    exporter = DatasetExporter(args.output)
    report_path = exporter.reports_dir / "duplicate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    return report


# ============================================================
# COMMAND: stats
# ============================================================

def cmd_stats(args):
    """Execute the 'stats' command."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Dataset Statistics")
    print("=" * 60)

    images = find_images(args.input)
    if not images:
        return None

    print(f"\n[INFO] Analyzing {len(images)} images in: {args.input}")
    print()

    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze(images)
    stats_dict = stats.to_dict()

    # Display stats
    print(f"  Total images:  {stats_dict['total_images']}")
    print(f"  Total size:    {stats_dict['total_size_mb']} MB")

    res = stats_dict.get("resolution", {})
    if res:
        w = res.get("width", {})
        h = res.get("height", {})
        print(f"\n  Resolution:")
        print(f"    Width:  {w.get('min',0)} - {w.get('max',0)} (avg: {w.get('mean',0)})")
        print(f"    Height: {h.get('min',0)} - {h.get('max',0)} (avg: {h.get('mean',0)})")

    fs = stats_dict.get("file_sizes", {})
    if fs:
        print(f"\n  File Sizes:")
        print(f"    Min: {fs.get('min_kb',0)} KB  |  Max: {fs.get('max_kb',0)} KB  |  Mean: {fs.get('mean_kb',0)} KB")

    fmts = stats_dict.get("formats", {})
    if fmts:
        print(f"\n  Formats:")
        for ext, count in fmts.items():
            print(f"    {ext}: {count}")

    colors = stats_dict.get("color_stats", {})
    if colors:
        print(f"\n  Color Channels:")
        for ch, d in colors.items():
            print(f"    {ch}: mean={d.get('mean_intensity',0)}, std={d.get('std_intensity',0)}")

    quality = stats_dict.get("quality_distribution", {})
    if quality:
        print(f"\n  Quality Distribution:")
        for cat, count in quality.items():
            print(f"    {cat}: {count}")

    print(f"\n{'=' * 60}")

    # Save stats JSON
    exporter = DatasetExporter(args.output)
    stats_path = exporter.reports_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"  Stats saved: {stats_path}")

    return stats_dict


# ============================================================
# COMMAND: pipeline
# ============================================================

def cmd_pipeline(args):
    """Execute the 'pipeline' command (augment + validate + stats + duplicates + HTML report)."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Full Pipeline")
    print("=" * 60)
    print()

    # Step 1: Augment
    augment_args = argparse.Namespace(
        input=args.input,
        output=args.output,
        count=args.count,
        transforms=args.transforms,
        seed=args.seed,
    )
    aug_log = cmd_augment(augment_args)

    # Step 2: Validate augmented output
    print(f"\n{'~' * 60}")
    print(f"  Step 2: Validating augmented images...")
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
    val_results, val_summary = cmd_validate(validate_args)

    # Step 3: Dataset statistics
    print(f"\n{'~' * 60}")
    print(f"  Step 3: Computing dataset statistics...")
    print(f"{'~' * 60}\n")

    stats_args = argparse.Namespace(
        input=augmented_dir,
        output=args.output,
    )
    stats_dict = cmd_stats(stats_args)

    # Step 4: Duplicate detection
    print(f"\n{'~' * 60}")
    print(f"  Step 4: Scanning for duplicates...")
    print(f"{'~' * 60}\n")

    compare_args = argparse.Namespace(
        input=augmented_dir,
        output=args.output,
        ssim_threshold=0.95,
        hash_threshold=5,
    )
    dup_report = cmd_compare(compare_args)

    # Step 5: Generate HTML report
    print(f"\n{'~' * 60}")
    print(f"  Step 5: Generating HTML report...")
    print(f"{'~' * 60}\n")

    if val_results and val_summary:
        exporter = DatasetExporter(args.output)
        html_path = exporter.generate_html_report(
            validation_results=val_results,
            summary=val_summary,
            dataset_stats=stats_dict,
            augmentation_log=aug_log,
            duplicate_report=dup_report,
        )
        print(f"  HTML report: {html_path}")

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")


# ============================================================
# COMMAND: report
# ============================================================

def cmd_report(args):
    """Execute the 'report' command (generate HTML from existing data)."""
    print("=" * 60)
    print("  CV Dataset Toolkit -- Generate HTML Report")
    print("=" * 60)

    exporter = DatasetExporter(args.output)

    # Load existing JSON data
    quality_json = exporter.reports_dir / "quality_report.json"
    stats_json = exporter.reports_dir / "dataset_stats.json"
    dup_json = exporter.reports_dir / "duplicate_report.json"
    aug_json = exporter.metadata_dir / "augmentation_log.json"

    if not quality_json.exists():
        print("[ERROR] No quality_report.json found. Run 'validate' or 'pipeline' first.")
        return

    with open(quality_json, "r") as f:
        quality_data = json.load(f)

    summary = quality_data.get("summary", {})

    # Create lightweight result objects for the HTML generator
    from toolkit.validator import QualityResult
    val_results = []
    for r in quality_data.get("results", []):
        qr = QualityResult(
            filename=r.get("filename", ""),
            passed=r.get("passed", False),
            blur_score=r.get("blur_score", 0),
            brightness=r.get("brightness", 0),
            contrast=r.get("contrast", 0),
            resolution=tuple(int(x) for x in r.get("resolution", "0x0").split("x")),
            aspect_ratio=r.get("aspect_ratio", 0),
            issues=r.get("issues", []),
        )
        val_results.append(qr)

    stats_dict = None
    if stats_json.exists():
        with open(stats_json, "r") as f:
            stats_dict = json.load(f)

    dup_report = None
    if dup_json.exists():
        with open(dup_json, "r") as f:
            dup_report = json.load(f)

    aug_log = None
    if aug_json.exists():
        with open(aug_json, "r") as f:
            aug_data = json.load(f)
            aug_log = aug_data.get("entries", [])

    html_path = exporter.generate_html_report(
        validation_results=val_results,
        summary=summary,
        dataset_stats=stats_dict,
        augmentation_log=aug_log,
        duplicate_report=dup_report,
    )

    print(f"\n  HTML report generated: {html_path}")
    print(f"{'=' * 60}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="CV Dataset Toolkit",
        description="Image augmentation, quality validation, and dataset analysis for AI/CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python main.py augment  --input ./images --output ./output --count 5\n"
               "  python main.py validate --input ./images --output ./output\n"
               "  python main.py compare  --input ./images --output ./output\n"
               "  python main.py stats    --input ./images --output ./output\n"
               "  python main.py pipeline --input ./images --output ./output --count 3\n"
               "  python main.py report   --output ./output\n"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Augment command ---
    aug_parser = subparsers.add_parser("augment", help="Augment images for dataset expansion")
    aug_parser.add_argument("--input", "-i", required=True, help="Input directory")
    aug_parser.add_argument("--output", "-o", required=True, help="Output directory")
    aug_parser.add_argument("--count", "-c", type=int, default=5, help="Variants per image (default: 5)")
    aug_parser.add_argument("--transforms", "-t", nargs="+", default=None,
                            choices=ImageAugmentor.AVAILABLE_TRANSFORMS,
                            help="Specific transforms (default: all 16)")
    aug_parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")

    # --- Validate command ---
    val_parser = subparsers.add_parser("validate", help="Validate image quality (8 checks)")
    val_parser.add_argument("--input", "-i", required=True, help="Input directory")
    val_parser.add_argument("--output", "-o", required=True, help="Output directory")
    val_parser.add_argument("--blur-threshold", type=float, default=100.0, help="Min blur score (default: 100.0)")
    val_parser.add_argument("--min-resolution", type=int, default=224, help="Min dimension px (default: 224)")
    val_parser.add_argument("--min-brightness", type=int, default=40, help="Min brightness (default: 40)")
    val_parser.add_argument("--max-brightness", type=int, default=220, help="Max brightness (default: 220)")

    # --- Compare command ---
    cmp_parser = subparsers.add_parser("compare", help="Find duplicate images")
    cmp_parser.add_argument("--input", "-i", required=True, help="Input directory")
    cmp_parser.add_argument("--output", "-o", required=True, help="Output directory")
    cmp_parser.add_argument("--ssim-threshold", type=float, default=0.95, help="SSIM duplicate threshold (default: 0.95)")
    cmp_parser.add_argument("--hash-threshold", type=int, default=5, help="Hash distance threshold (default: 5)")

    # --- Stats command ---
    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics")
    stats_parser.add_argument("--input", "-i", required=True, help="Input directory")
    stats_parser.add_argument("--output", "-o", required=True, help="Output directory")

    # --- Pipeline command ---
    pipe_parser = subparsers.add_parser("pipeline", help="Full pipeline: augment + validate + stats + duplicates + report")
    pipe_parser.add_argument("--input", "-i", required=True, help="Input directory")
    pipe_parser.add_argument("--output", "-o", required=True, help="Output directory")
    pipe_parser.add_argument("--count", "-c", type=int, default=5, help="Variants per image (default: 5)")
    pipe_parser.add_argument("--transforms", "-t", nargs="+", default=None,
                              choices=ImageAugmentor.AVAILABLE_TRANSFORMS, help="Specific transforms")
    pipe_parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    pipe_parser.add_argument("--blur-threshold", type=float, default=100.0, help="Min blur score")
    pipe_parser.add_argument("--min-resolution", type=int, default=224, help="Min dimension px")

    # --- Report command ---
    rpt_parser = subparsers.add_parser("report", help="Generate HTML report from existing data")
    rpt_parser.add_argument("--output", "-o", required=True, help="Output directory (with existing reports)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "augment": cmd_augment,
        "validate": cmd_validate,
        "compare": cmd_compare,
        "stats": cmd_stats,
        "pipeline": cmd_pipeline,
        "report": cmd_report,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
