"""
Floor Plan Preprocessing Module
Image preprocessing for raster plans: deskew, denoise, adaptive threshold.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    # Deskew
    enable_deskew: bool = True
    deskew_angle_threshold: float = 5.0  # degrees

    # Denoise
    enable_denoise: bool = True
    denoise_strength: int = 10

    # Contrast enhancement
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)

    # Binarization
    adaptive_block_size: int = 35
    adaptive_c: int = 10

    # Morphological cleanup
    remove_small_noise: bool = True
    noise_kernel_size: int = 3

    # Border handling
    add_border: bool = True
    border_size: int = 50


@dataclass
class PreprocessResult:
    """Result of preprocessing."""
    original: np.ndarray
    grayscale: np.ndarray
    binarized: np.ndarray
    cleaned: np.ndarray
    deskew_angle: float = 0.0
    scale_factor: float = 1.0


class Preprocessor:
    """
    Image preprocessing for floor plans.
    Handles deskew, denoise, and binarization.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def process(self, image: np.ndarray) -> PreprocessResult:
        """
        Full preprocessing pipeline.

        Args:
            image: Input BGR image

        Returns:
            PreprocessResult with all intermediate images
        """
        logger.info(f"Preprocessing image of shape {image.shape}")

        original = image.copy()

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Deskew
        deskew_angle = 0.0
        if self.config.enable_deskew:
            gray, deskew_angle = self._deskew(gray)
            if abs(deskew_angle) > 0.1:
                logger.info(f"Deskewed by {deskew_angle:.2f} degrees")

        # Denoise
        if self.config.enable_denoise:
            gray = cv2.fastNlMeansDenoising(
                gray,
                h=self.config.denoise_strength
            )

        # Contrast enhancement (CLAHE)
        if self.config.enable_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size
            )
            gray = clahe.apply(gray)

        # Add border if needed
        if self.config.add_border:
            border = self.config.border_size
            gray = cv2.copyMakeBorder(
                gray, border, border, border, border,
                cv2.BORDER_CONSTANT, value=255
            )

        # Binarization - adaptive thresholding
        binarized = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )

        # Cleanup small noise
        cleaned = binarized.copy()
        if self.config.remove_small_noise:
            cleaned = self._remove_small_components(
                cleaned,
                min_size=self.config.noise_kernel_size ** 2 * 4
            )

        return PreprocessResult(
            original=original,
            grayscale=gray,
            binarized=binarized,
            cleaned=cleaned,
            deskew_angle=deskew_angle
        )

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew in image.

        Returns:
            Tuple of (deskewed image, angle in degrees)
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) < 10:
            return image, 0.0

        # Calculate angles of lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Normalize to -90 to 90
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180

            # Only consider nearly horizontal or vertical lines
            if abs(angle) < 45:
                angles.append(angle)
            elif abs(angle) > 45:
                # Vertical line - compute deviation from 90
                angles.append(angle - 90 if angle > 0 else angle + 90)

        if not angles:
            return image, 0.0

        # Use median angle
        median_angle = np.median(angles)

        # Only deskew if angle is significant but not too large
        if abs(median_angle) < 0.1 or abs(median_angle) > self.config.deskew_angle_threshold:
            return image, 0.0

        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calculate new image size to avoid cropping
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        return rotated, median_angle

    def _remove_small_components(self, binary: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small connected components (noise)."""
        # Label connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # Create output image
        output = np.zeros_like(binary)

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output[labels == i] = 255

        return output


def create_wall_mask(binary: np.ndarray, wall_thickness: int = 5) -> np.ndarray:
    """
    Create a wall mask by detecting and thickening line elements.

    Args:
        binary: Binary image (white lines on black)
        wall_thickness: Thickness for dilation

    Returns:
        Wall mask image
    """
    # Morphological operations to extract walls
    # Horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Vertical kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Combine
    walls = cv2.bitwise_or(horizontal, vertical)

    # Dilate to thicken walls
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (wall_thickness, wall_thickness)
    )
    walls = cv2.dilate(walls, dilate_kernel, iterations=1)

    return walls


def close_gaps(binary: np.ndarray, gap_size: int = 10) -> np.ndarray:
    """
    Close small gaps in walls (e.g., door openings).

    Args:
        binary: Binary wall mask
        gap_size: Maximum gap to close

    Returns:
        Gap-closed mask
    """
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, gap_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


def fill_holes(binary: np.ndarray, max_hole_area: int = 500) -> np.ndarray:
    """
    Fill small holes in regions (e.g., columns, fixtures).

    Args:
        binary: Binary mask
        max_hole_area: Maximum hole area to fill

    Returns:
        Hole-filled mask
    """
    # Invert
    inverted = cv2.bitwise_not(binary)

    # Find contours of holes
    contours, _ = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Fill small holes
    output = binary.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max_hole_area:
            cv2.drawContours(output, [contour], -1, 255, -1)

    return output


def preprocess_plan(image: np.ndarray, config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    """
    Convenience function to preprocess a floor plan image.

    Args:
        image: Input BGR image
        config: Optional preprocessing configuration

    Returns:
        PreprocessResult
    """
    preprocessor = Preprocessor(config)
    return preprocessor.process(image)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = preprocess_plan(img)
            print(f"Original shape: {result.original.shape}")
            print(f"Processed shape: {result.cleaned.shape}")
            print(f"Deskew angle: {result.deskew_angle:.2f}")

            # Save outputs
            cv2.imwrite("preprocessed_gray.png", result.grayscale)
            cv2.imwrite("preprocessed_binary.png", result.binarized)
            cv2.imwrite("preprocessed_cleaned.png", result.cleaned)
            print("Saved preprocessed images")
    else:
        print("Usage: python preprocess.py <image_file>")
