"""
Lerp — Image Utilities
======================
Pre-processing for raster images before vectorization:
- Background removal (ensure 100% transparent bg)
- Color quantization (reduce to N discrete colors)
- Contrast enhancement
"""

from pathlib import Path
from PIL import Image
import numpy as np


def remove_background(input_path: Path, output_path: Path = None,
                      threshold: int = 240) -> Path:
    """
    Remove near-white backgrounds by converting to transparent.
    
    Simple threshold-based approach for Recraft output which already
    has clean white backgrounds. For complex images, use rembg or
    remove.bg API instead.
    """
    output_path = output_path or input_path.with_stem(input_path.stem + "_nobg")

    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Find pixels where R, G, B are all above threshold (near white)
    white_mask = np.all(data[:, :, :3] > threshold, axis=2)
    data[white_mask] = [255, 255, 255, 0]  # Make transparent

    result = Image.fromarray(data)
    result.save(output_path)
    return output_path


def quantize_colors(input_path: Path, n_colors: int = 4,
                    output_path: Path = None) -> Path:
    """
    Reduce image to exactly N discrete colors.
    
    Critical pre-processing step: even Recraft sometimes outputs subtle
    gradients or anti-aliasing. Quantizing to N colors gives vtracer
    much cleaner input and dramatically fewer spurious paths.
    """
    output_path = output_path or input_path.with_stem(input_path.stem + "_quantized")

    img = Image.open(input_path).convert("RGB")
    quantized = img.quantize(colors=n_colors, method=Image.MEDIANCUT)

    # Get actual color count for logging
    colors = quantized.convert("RGB").getcolors(maxcolors=256)

    quantized.convert("RGB").save(output_path)

    return output_path


def quantize_with_palette(input_path: Path, palette_hex: list[str],
                          output_path: Path = None) -> Path:
    """
    Quantize image to a specific color palette (from design spec).
    
    Maps each pixel to the nearest color in the provided palette.
    This ensures the traced SVG uses exactly the spec's hex colors.
    """
    output_path = output_path or input_path.with_stem(input_path.stem + "_palettized")

    img = Image.open(input_path).convert("RGB")
    data = np.array(img, dtype=np.float32)

    # Parse hex colors to RGB
    palette_rgb = []
    for hex_color in palette_hex:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        palette_rgb.append([r, g, b])
    palette_rgb = np.array(palette_rgb, dtype=np.float32)

    # Reshape for broadcasting: (H, W, 1, 3) vs (1, 1, N, 3)
    h, w, _ = data.shape
    pixels = data.reshape(h, w, 1, 3)
    colors = palette_rgb.reshape(1, 1, -1, 3)

    # Find nearest palette color for each pixel (Euclidean distance)
    distances = np.sqrt(np.sum((pixels - colors) ** 2, axis=3))
    nearest_indices = np.argmin(distances, axis=2)

    # Map to palette
    result = palette_rgb[nearest_indices].astype(np.uint8)

    Image.fromarray(result).save(output_path)
    return output_path


def prepare_for_tracing(input_path: Path, n_colors: int = 4,
                        palette_hex: list[str] = None) -> Path:
    """
    Full pre-processing pipeline: bg removal → quantization.
    
    Returns path to the processed image ready for vtracer.
    """
    # Step 1: Remove background
    nobg_path = remove_background(input_path)

    # Step 2: Quantize
    if palette_hex:
        final_path = quantize_with_palette(nobg_path, palette_hex)
    else:
        final_path = quantize_colors(nobg_path, n_colors)

    return final_path
