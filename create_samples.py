"""
Generate sample test images to demonstrate the toolkit.
Creates synthetic images with known properties for testing.
"""

import cv2
import numpy as np
from pathlib import Path


def create_sample_images(output_dir: str = "sample_images"):
    """Generate sample images with different characteristics for testing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. A sharp, well-lit gradient image with shapes
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Blue-to-green gradient background
    for x in range(640):
        img1[:, x] = [255 - int(x * 0.4), int(x * 0.3), int(x * 0.15)]
    # Add geometric shapes
    cv2.rectangle(img1, (50, 50), (200, 200), (0, 255, 255), 3)
    cv2.circle(img1, (400, 240), 100, (255, 100, 50), -1)
    cv2.putText(img1, "SHARP", (220, 300), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3)
    cv2.imwrite(str(out / "sample_sharp.jpg"), img1)
    print(f"  Created: sample_sharp.jpg (640x480, good quality)")

    # 2. A blurry image (will fail blur check)
    img2 = cv2.GaussianBlur(img1, (31, 31), 0)
    cv2.imwrite(str(out / "sample_blurry.jpg"), img2)
    print(f"  Created: sample_blurry.jpg (640x480, blurry)")

    # 3. A dark image (will fail brightness check)
    img3 = (img1 * 0.15).astype(np.uint8)
    cv2.imwrite(str(out / "sample_dark.jpg"), img3)
    print(f"  Created: sample_dark.jpg (640x480, too dark)")

    # 4. A low-resolution image (will fail resolution check)
    img4 = cv2.resize(img1, (100, 75))
    cv2.imwrite(str(out / "sample_lowres.jpg"), img4)
    print(f"  Created: sample_lowres.jpg (100x75, low resolution)")

    # 5. A high-quality natural-looking scene
    img5 = np.zeros((600, 800, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(300):
        img5[y, :] = [230 - int(y * 0.3), 180 - int(y * 0.2), 50 + int(y * 0.5)]
    # Ground
    for y in range(300, 600):
        img5[y, :] = [40, 120 + int((y - 300) * 0.1), 60]
    # Sun
    cv2.circle(img5, (600, 100), 50, (50, 200, 255), -1)
    # Trees
    for tx in [150, 350, 550]:
        cv2.rectangle(img5, (tx - 10, 250), (tx + 10, 350), (20, 60, 30), -1)
        pts = np.array([[tx - 60, 280], [tx + 60, 280], [tx, 180]], np.int32)
        cv2.fillPoly(img5, [pts], (30, 100, 20))
    cv2.imwrite(str(out / "sample_scene.jpg"), img5)
    print(f"  Created: sample_scene.jpg (800x600, good quality)")

    print(f"\n  Total: 5 sample images created in '{output_dir}/'")


if __name__ == "__main__":
    create_sample_images()
