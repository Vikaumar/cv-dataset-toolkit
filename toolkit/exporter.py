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

    def generate_html_report(self, validation_results: List[Any],
                              summary: Dict[str, Any],
                              dataset_stats: Dict[str, Any] = None,
                              augmentation_log: List[Dict] = None,
                              duplicate_report: Dict[str, Any] = None) -> str:
        """
        Generate a self-contained HTML report with all pipeline results.

        Args:
            validation_results: List of QualityResult objects.
            summary: Validation summary dict.
            dataset_stats: Optional DatasetStats.to_dict() output.
            augmentation_log: Optional list of augmentation metadata entries.
            duplicate_report: Optional duplicate finder results.

        Returns:
            Path to saved HTML report.
        """
        report_path = self.reports_dir / "report.html"

        html_parts = [self._html_header()]

        # Summary section
        html_parts.append(self._html_summary(summary))

        # Dataset statistics section
        if dataset_stats:
            html_parts.append(self._html_dataset_stats(dataset_stats))

        # Quality results table
        html_parts.append(self._html_quality_table(validation_results))

        # Duplicate report
        if duplicate_report:
            html_parts.append(self._html_duplicates(duplicate_report))

        # Augmentation log
        if augmentation_log:
            html_parts.append(self._html_augmentation_log(augmentation_log))

        html_parts.append("</div></body></html>")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        return str(report_path)

    def _html_header(self) -> str:
        """Generate HTML header with inline CSS."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CV Dataset Toolkit - Pipeline Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 28px; margin-bottom: 8px; color: #f8fafc; }
  h2 { font-size: 20px; margin: 32px 0 16px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }
  .subtitle { color: #64748b; font-size: 14px; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 8px; padding: 20px; border: 1px solid #334155; }
  .card-value { font-size: 32px; font-weight: 700; color: #f8fafc; }
  .card-label { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  .pass { color: #4ade80; }
  .fail { color: #f87171; }
  .warn { color: #fbbf24; }
  table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }
  th { background: #1e293b; color: #94a3b8; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #334155; }
  td { padding: 8px 12px; border-bottom: 1px solid #1e293b; }
  tr:hover { background: #1e293b; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-pass { background: #064e3b; color: #4ade80; }
  .badge-fail { background: #450a0a; color: #f87171; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }
  .stat-box { background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }
  .stat-box h3 { font-size: 14px; color: #94a3b8; margin-bottom: 8px; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
  .stat-key { color: #64748b; }
  .stat-val { color: #e2e8f0; font-weight: 500; }
  .bar-container { height: 8px; background: #334155; border-radius: 4px; margin-top: 8px; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .issue-list { list-style: none; }
  .issue-list li { padding: 6px 0; font-size: 13px; color: #fbbf24; }
  .issue-list li::before { content: "! "; font-weight: bold; }
</style>
</head>
<body>
<div class="container">
<h1>CV Dataset Toolkit Report</h1>
<p class="subtitle">Generated by CV Dataset Toolkit v1.0.0</p>"""

    def _html_summary(self, summary: Dict[str, Any]) -> str:
        """Generate summary cards section."""
        pass_rate = summary.get("pass_rate", 0)
        rate_class = "pass" if pass_rate >= 80 else ("warn" if pass_rate >= 50 else "fail")

        return f"""
<h2>Validation Summary</h2>
<div class="cards">
  <div class="card">
    <div class="card-value">{summary.get('total_images', 0)}</div>
    <div class="card-label">Total Images</div>
  </div>
  <div class="card">
    <div class="card-value pass">{summary.get('passed', 0)}</div>
    <div class="card-label">Passed</div>
  </div>
  <div class="card">
    <div class="card-value fail">{summary.get('failed', 0)}</div>
    <div class="card-label">Failed</div>
  </div>
  <div class="card">
    <div class="card-value {rate_class}">{pass_rate}%</div>
    <div class="card-label">Pass Rate</div>
  </div>
  <div class="card">
    <div class="card-value">{summary.get('avg_blur_score', 0)}</div>
    <div class="card-label">Avg Blur Score</div>
  </div>
  <div class="card">
    <div class="card-value">{summary.get('avg_brightness', 0)}</div>
    <div class="card-label">Avg Brightness</div>
  </div>
</div>"""

    def _html_dataset_stats(self, stats: Dict[str, Any]) -> str:
        """Generate dataset statistics section."""
        resolution = stats.get("resolution", {})
        file_sizes = stats.get("file_sizes", {})
        formats = stats.get("formats", {})
        color = stats.get("color_stats", {})
        quality = stats.get("quality_distribution", {})

        res_html = ""
        if resolution:
            w = resolution.get("width", {})
            h = resolution.get("height", {})
            res_html = f"""
<div class="stat-box">
  <h3>Resolution</h3>
  <div class="stat-row"><span class="stat-key">Width</span><span class="stat-val">{w.get('min',0)} - {w.get('max',0)} (avg: {w.get('mean',0)})</span></div>
  <div class="stat-row"><span class="stat-key">Height</span><span class="stat-val">{h.get('min',0)} - {h.get('max',0)} (avg: {h.get('mean',0)})</span></div>
  <div class="stat-row"><span class="stat-key">Avg Pixels</span><span class="stat-val">{resolution.get('total_pixels_avg',0):,}</span></div>
</div>"""

        size_html = ""
        if file_sizes:
            size_html = f"""
<div class="stat-box">
  <h3>File Sizes</h3>
  <div class="stat-row"><span class="stat-key">Min</span><span class="stat-val">{file_sizes.get('min_kb',0)} KB</span></div>
  <div class="stat-row"><span class="stat-key">Max</span><span class="stat-val">{file_sizes.get('max_kb',0)} KB</span></div>
  <div class="stat-row"><span class="stat-key">Mean</span><span class="stat-val">{file_sizes.get('mean_kb',0)} KB</span></div>
  <div class="stat-row"><span class="stat-key">Total</span><span class="stat-val">{stats.get('total_size_mb',0)} MB</span></div>
</div>"""

        fmt_html = ""
        if formats:
            fmt_rows = "".join(
                f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>'
                for k, v in formats.items()
            )
            fmt_html = f'<div class="stat-box"><h3>Formats</h3>{fmt_rows}</div>'

        color_html = ""
        if color:
            color_rows = "".join(
                f'<div class="stat-row"><span class="stat-key">{ch}</span><span class="stat-val">mean: {d.get("mean_intensity",0)}, std: {d.get("std_intensity",0)}</span></div>'
                for ch, d in color.items()
            )
            color_html = f'<div class="stat-box"><h3>Color Channels</h3>{color_rows}</div>'

        quality_html = ""
        if quality:
            q_rows = "".join(
                f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>'
                for k, v in quality.items()
            )
            quality_html = f'<div class="stat-box"><h3>Quality Distribution</h3>{q_rows}</div>'

        return f"""
<h2>Dataset Statistics</h2>
<div class="stat-grid">
  {res_html}
  {size_html}
  {fmt_html}
  {color_html}
  {quality_html}
</div>"""

    def _html_quality_table(self, results: List[Any]) -> str:
        """Generate quality results table."""
        rows = ""
        for r in results:
            status = '<span class="badge badge-pass">PASS</span>' if r.passed else '<span class="badge badge-fail">FAIL</span>'
            issues = "; ".join(r.issues) if r.issues else "-"
            rows += f"""<tr>
  <td>{r.filename}</td>
  <td>{status}</td>
  <td>{r.blur_score}</td>
  <td>{r.brightness}</td>
  <td>{r.contrast}</td>
  <td>{r.resolution[0]}x{r.resolution[1]}</td>
  <td>{r.aspect_ratio}</td>
  <td style="font-size:11px;color:#94a3b8;">{issues}</td>
</tr>"""

        return f"""
<h2>Quality Results ({len(results)} images)</h2>
<div style="overflow-x:auto;">
<table>
<thead><tr>
  <th>Filename</th><th>Status</th><th>Blur</th><th>Brightness</th>
  <th>Contrast</th><th>Resolution</th><th>AR</th><th>Issues</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>"""

    def _html_duplicates(self, report: Dict[str, Any]) -> str:
        """Generate duplicate detection section."""
        dupes = report.get("duplicates", [])
        rows = ""
        for d in dupes:
            rows += f"""<tr>
  <td>{d['image_a']}</td>
  <td>{d['image_b']}</td>
  <td>{d.get('ssim', 'N/A')}</td>
  <td>{d.get('hash_distance', 'N/A')}</td>
</tr>"""

        return f"""
<h2>Duplicate Detection</h2>
<div class="cards">
  <div class="card"><div class="card-value">{report.get('total_images',0)}</div><div class="card-label">Total</div></div>
  <div class="card"><div class="card-value pass">{report.get('unique_images',0)}</div><div class="card-label">Unique</div></div>
  <div class="card"><div class="card-value warn">{report.get('duplicate_pairs',0)}</div><div class="card-label">Duplicate Pairs</div></div>
</div>
<table>
<thead><tr><th>Image A</th><th>Image B</th><th>SSIM</th><th>Hash Distance</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""

    def _html_augmentation_log(self, log: List[Dict]) -> str:
        """Generate augmentation log section."""
        rows = ""
        for entry in log[:50]:  # Limit to 50 entries
            transform = entry.get("transform", "unknown")
            source = entry.get("source_image", "")
            params = {k: v for k, v in entry.items()
                      if k not in ("transform", "source_image", "output_path")}
            param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "-"
            rows += f"<tr><td>{source}</td><td>{transform}</td><td style='font-size:11px;'>{param_str}</td></tr>"

        shown = min(len(log), 50)
        total = len(log)
        note = f" (showing {shown} of {total})" if total > 50 else ""

        return f"""
<h2>Augmentation Log{note}</h2>
<table>
<thead><tr><th>Source</th><th>Transform</th><th>Parameters</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""

