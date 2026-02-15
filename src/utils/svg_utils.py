"""
Lerp — SVG Utilities
====================
SVG analysis, metrics, and rendering utilities used across the pipeline.
"""

import re
import math
from pathlib import Path
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def count_nodes(svg_path: Path) -> dict:
    """Count anchor points and path commands in an SVG."""
    try:
        from svgpathtools import svg2paths
        paths, attrs = svg2paths(str(svg_path))
        total_segments = sum(len(path) for path in paths)
        return {"paths": len(paths), "nodes": total_segments}
    except Exception as e:
        return {"paths": 0, "nodes": 0, "error": str(e)}


def file_size(svg_path: Path) -> int:
    """Get SVG file size in bytes."""
    return svg_path.stat().st_size


def symmetry_score(svg_path: Path) -> dict:
    """
    Calculate bilateral symmetry using Hausdorff distance.
    
    Returns:
        {score: float, verdict: str}
        Lower score = more symmetric.
        < 0.05 = perfect, < 0.15 = acceptable, > 0.15 = lopsided
    """
    try:
        from svgpathtools import svg2paths
        paths, _ = svg2paths(str(svg_path))
        if not paths:
            return {"score": None, "verdict": "No paths"}

        # Sample points from all paths
        points = []
        for path in paths:
            for i in range(50):
                t = i / 50
                try:
                    pt = path.point(t)
                    points.append([pt.real, pt.imag])
                except:
                    continue

        if len(points) < 10:
            return {"score": None, "verdict": "Too few points"}

        points = np.array(points)
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2

        # Mirror
        mirrored = points.copy()
        mirrored[:, 0] = 2 * center_x - mirrored[:, 0]

        # Hausdorff distance normalized by bounding box diagonal
        bbox_diag = np.sqrt(
            (points[:, 0].max() - points[:, 0].min()) ** 2 +
            (points[:, 1].max() - points[:, 1].min()) ** 2
        )

        h_dist = max(
            directed_hausdorff(points, mirrored)[0],
            directed_hausdorff(mirrored, points)[0],
        )

        score = h_dist / bbox_diag if bbox_diag > 0 else 0

        if score < 0.05:
            verdict = "✅ Perfectly symmetric"
        elif score < 0.15:
            verdict = "⚠️ Minor asymmetry"
        else:
            verdict = "❌ Lopsided"

        return {"score": round(score, 4), "verdict": verdict}

    except Exception as e:
        return {"score": None, "verdict": f"Error: {e}"}


def contrast_ratio(hex1: str, hex2: str) -> float:
    """
    Calculate WCAG contrast ratio between two hex colors.
    WCAG AA minimum: 4.5:1
    """
    def hex_to_luminance(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        # sRGB to linear
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = hex_to_luminance(hex1)
    l2 = hex_to_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return round((lighter + 0.05) / (darker + 0.05), 2)


def balance_index(svg_path: Path) -> dict:
    """
    Calculate visual balance: center of mass vs geometric center.
    
    Target: < 0.02 (2% of canvas diagonal)
    """
    try:
        from svgpathtools import svg2paths
        paths, _ = svg2paths(str(svg_path))
        if not paths:
            return {"score": None, "verdict": "No paths"}

        # Sample weighted points
        points = []
        for path in paths:
            path_len = path.length()
            n_samples = max(5, int(path_len / 10))
            for i in range(n_samples):
                t = i / n_samples
                try:
                    pt = path.point(t)
                    points.append([pt.real, pt.imag])
                except:
                    continue

        if not points:
            return {"score": None, "verdict": "No points"}

        points = np.array(points)

        # Geometric center (bounding box)
        geo_center = np.array([
            (points[:, 0].min() + points[:, 0].max()) / 2,
            (points[:, 1].min() + points[:, 1].max()) / 2,
        ])

        # Visual center of mass
        vis_center = np.array([points[:, 0].mean(), points[:, 1].mean()])

        # Canvas diagonal
        diag = np.sqrt(
            (points[:, 0].max() - points[:, 0].min()) ** 2 +
            (points[:, 1].max() - points[:, 1].min()) ** 2
        )

        score = np.linalg.norm(geo_center - vis_center) / diag if diag > 0 else 0

        if score < 0.02:
            verdict = "✅ Well balanced"
        elif score < 0.05:
            verdict = "⚠️ Slightly off-center"
        else:
            verdict = "❌ Unbalanced"

        return {"score": round(score, 4), "verdict": verdict}

    except Exception as e:
        return {"score": None, "verdict": f"Error: {e}"}


def render_at_sizes(svg_path: Path, sizes: list[int] = None,
                    output_dir: Path = None) -> list[Path]:
    """Render SVG at multiple sizes for multi-scale QA."""
    import cairosvg

    sizes = sizes or [16, 32, 64, 128, 256, 512]
    output_dir = output_dir or svg_path.parent

    renders = []
    for s in sizes:
        output = output_dir / f"{svg_path.stem}_{s}px.png"
        try:
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(output),
                output_width=s,
                output_height=s,
            )
            renders.append(output)
        except Exception as e:
            print(f"⚠️ Render failed at {s}px: {e}")

    return renders


def full_evaluation(svg_path: Path) -> dict:
    """Run all evaluation metrics on an SVG."""
    return {
        "file_size": file_size(svg_path),
        "nodes": count_nodes(svg_path),
        "symmetry": symmetry_score(svg_path),
        "balance": balance_index(svg_path),
    }
