"""
Lerp — Deterministic Shape Detector
=====================================
Classifies SVG paths into geometric primitives using pure math.
No LLM calls. Zero cost. Zero hallucination.

Conservative mode (default): only converts shapes it's highly confident
about. Everything else keeps original path data — visually perfect output
with partial size reduction.

Pipeline position: Stage 3b, Pass 0 (before LLM snap engine)
  1. ShapeDetector converts obvious shapes (rects, ellipses, verified polygons)
  2. Remaining complex paths go to LLM for path-by-path cleanup (optional)
  3. SVGO final minification

Usage:
    from src.utils.shape_detector import ShapeDetector

    detector = ShapeDetector()
    result = detector.process("bloated_icon.svg", "cleaned_icon.svg")

    print(f"Converted {result.converted}/{result.total_paths} paths")
    print(f"Size: {result.input_size_bytes:,}B → {result.output_size_bytes:,}B")
    print(f"Complex paths for LLM: {result.complex_indices}")
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from svgpathtools import svg2paths


# ─── Result Dataclass ─────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Result from the shape detector."""
    input_path: Path
    output_path: Path
    total_paths: int
    converted: int
    kept: int
    complex_indices: list[int]
    input_size_bytes: int
    output_size_bytes: int
    size_reduction_pct: float
    classifications: list[dict] = field(default_factory=list)


# ─── Shape Detector ───────────────────────────────────────────────────

class ShapeDetector:
    """
    Deterministic SVG path → primitive converter.

    Conservative by default: only converts shapes where the polygon
    approximation is verified to be faithful (< 2.5% max deviation
    from original sampled points). Everything else keeps original
    path data untouched.

    Usage:
        detector = ShapeDetector()
        result = detector.process("input.svg", "output.svg")

        # Complex paths that weren't converted (for optional LLM pass)
        complex_paths = detector.get_complex_paths("input.svg")
    """

    def __init__(
        self,
        n_samples: int = 1000,
        corner_window: int = 8,
        corner_threshold_deg: float = 25.0,
        corner_merge_distance: int = 30,
        ellipse_error_threshold: float = 0.08,
        polygon_deviation_pct: float = 2.5,
        max_polygon_vertices: int = 20,
        min_shape_size: float = 15.0,
        normalize_colors: bool = True,
    ):
        """
        Args:
            n_samples: Points to sample along each path
            corner_window: Window size for corner detection (larger = smoother)
            corner_threshold_deg: Min angle change to detect a corner
            corner_merge_distance: Min sample distance between distinct corners
            ellipse_error_threshold: Max error for ellipse fit (lower = stricter)
            polygon_deviation_pct: Max % deviation from original to accept polygon
            max_polygon_vertices: Max vertices before keeping as original
            min_shape_size: Skip shapes smaller than this (px) — not worth the risk
            normalize_colors: Normalize near-black/white colors
        """
        self.n_samples = n_samples
        self.corner_window = corner_window
        self.corner_threshold = np.radians(corner_threshold_deg)
        self.corner_merge_distance = corner_merge_distance
        self.ellipse_threshold = ellipse_error_threshold
        self.polygon_deviation_pct = polygon_deviation_pct
        self.max_polygon_vertices = max_polygon_vertices
        self.min_shape_size = min_shape_size
        self.normalize_colors = normalize_colors

    # ── Main Entry Point ──────────────────────────────────────────

    def process(self, input_svg: Path | str, output_svg: Path | str = None) -> DetectorResult:
        """
        Process an SVG file: classify paths and output clean primitives.
        Paths that can't be confidently converted keep original data.
        """
        input_svg = Path(input_svg)
        output_svg = Path(output_svg) if output_svg else input_svg.with_stem(
            input_svg.stem + "_detected"
        )

        paths, attrs = svg2paths(str(input_svg))
        viewbox = self._extract_viewbox(input_svg)

        svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">']
        classifications = []
        converted = 0
        kept = 0
        complex_indices = []

        for i, (path, attr) in enumerate(zip(paths, attrs)):
            fill = attr.get("fill", "none")
            if self.normalize_colors:
                fill = self._normalize_color(fill)

            shape_type, svg_code = self._classify(path, fill)

            classifications.append(
                {"index": i, "type": shape_type, "fill": fill, "segments": len(path)}
            )

            if svg_code:
                converted += 1
                svg_lines.append(f"  {svg_code}")
            else:
                kept += 1
                complex_indices.append(i)
                d = attr.get("d", "")
                svg_lines.append(f'  <path fill="{fill}" d="{d}"/>')

        svg_lines.append("</svg>")
        svg_output = "\n".join(svg_lines)

        output_svg.parent.mkdir(parents=True, exist_ok=True)
        output_svg.write_text(svg_output)

        input_size = input_svg.stat().st_size
        output_size = len(svg_output.encode())

        return DetectorResult(
            input_path=input_svg,
            output_path=output_svg,
            total_paths=len(paths),
            converted=converted,
            kept=kept,
            complex_indices=complex_indices,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            size_reduction_pct=round(
                (1 - output_size / max(input_size, 1)) * 100, 1
            ),
            classifications=classifications,
        )

    # ── Classification ────────────────────────────────────────────

    def _classify(self, path, fill: str) -> tuple[str, str | None]:
        """
        Classify a single path. Returns (type_name, svg_code_or_None).
        None means "keep original path data".
        """
        seg_types = [type(seg).__name__ for seg in path]

        if not path.isclosed():
            return "keep", None

        # ── Exact rectangle (4 line segments — trivially correct) ──
        if all(t == "Line" for t in seg_types) and len(path) == 4:
            xmin, xmax, ymin, ymax = path.bbox()
            code = (
                f'<rect x="{xmin:.0f}" y="{ymin:.0f}" '
                f'width="{xmax - xmin:.0f}" height="{ymax - ymin:.0f}" fill="{fill}"/>'
            )
            return "rect", code

        # ── Sample points ──
        pts = self._sample_points(path)
        if pts is None or len(pts) < 20:
            return "keep", None

        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()

        # Skip tiny shapes — not worth the risk of distortion
        if w < self.min_shape_size or h < self.min_shape_size:
            return "keep", None

        # ── Ellipse / Circle (tight threshold) ──
        ellipse = self._fit_ellipse(pts)
        if ellipse:
            cx, cy, rx, ry, err = ellipse
            if err < self.ellipse_threshold:
                aspect = rx / ry if ry > 0 else 999
                if 0.9 < aspect < 1.1:
                    r = (rx + ry) / 2
                    return "circle", (
                        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" '
                        f'r="{r:.0f}" fill="{fill}"/>'
                    )
                else:
                    return "ellipse", (
                        f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" '
                        f'rx="{rx:.0f}" ry="{ry:.0f}" fill="{fill}"/>'
                    )

        # ── Polygon (only if verified faithful) ──
        vertices = self._extract_vertices(pts)
        if vertices is not None and 3 <= len(vertices) <= self.max_polygon_vertices:
            if self._verify_polygon(pts, vertices):
                points_str = " ".join(
                    f"{v[0]:.0f},{v[1]:.0f}" for v in vertices
                )
                return "polygon", f'<polygon points="{points_str}" fill="{fill}"/>'

        return "keep", None

    # ── Geometry Helpers ──────────────────────────────────────────

    def _sample_points(self, path) -> np.ndarray | None:
        """Sample evenly-spaced points along a path."""
        pts = []
        for i in range(self.n_samples):
            t = i / self.n_samples
            try:
                pt = path.point(t)
                pts.append([pt.real, pt.imag])
            except Exception:
                continue
        return np.array(pts) if len(pts) >= 20 else None

    def _fit_ellipse(self, pts: np.ndarray) -> tuple | None:
        """
        Fit ellipse via bounding box method.
        Returns (cx, cy, rx, ry, error) or None.
        Error = std deviation of (x-cx)²/rx² + (y-cy)²/ry² from 1.0
        """
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        rx, ry = (xmax - xmin) / 2, (ymax - ymin) / 2

        if rx < 2 or ry < 2:
            return None

        vals = ((pts[:, 0] - cx) / rx) ** 2 + ((pts[:, 1] - cy) / ry) ** 2
        error = float(np.std(vals - 1.0))
        return cx, cy, rx, ry, error

    def _extract_vertices(self, pts: np.ndarray) -> np.ndarray | None:
        """
        Find corner vertices by detecting direction changes.
        Uses a sliding window for noise robustness.
        """
        w = self.corner_window
        if len(pts) < w * 3:
            return None

        corners = []
        for i in range(w, len(pts) - w):
            d_before = pts[i] - pts[i - w]
            d_after = pts[i + w] - pts[i]
            n1, n2 = np.linalg.norm(d_before), np.linalg.norm(d_after)
            if n1 < 1e-10 or n2 < 1e-10:
                continue
            cos_a = np.clip(np.dot(d_before, d_after) / (n1 * n2), -1, 1)
            angle = np.arccos(cos_a)
            if angle > self.corner_threshold:
                corners.append((i, angle, pts[i]))

        if not corners:
            return None

        # Merge nearby corners (same physical corner)
        merged = []
        group = [corners[0]]
        for c in corners[1:]:
            if c[0] - group[-1][0] < self.corner_merge_distance:
                group.append(c)
            else:
                best = max(group, key=lambda x: x[1])
                merged.append(best[2])
                group = [c]
        merged.append(max(group, key=lambda x: x[1])[2])

        return np.array(merged) if len(merged) >= 3 else None

    def _verify_polygon(self, pts: np.ndarray, vertices: np.ndarray) -> bool:
        """
        Verify polygon faithfulness: check that sampled points don't
        deviate more than polygon_deviation_pct from polygon edges.
        This is the key guard against visual artifacts.
        """
        bbox_diag = np.sqrt(
            (pts[:, 0].max() - pts[:, 0].min()) ** 2
            + (pts[:, 1].max() - pts[:, 1].min()) ** 2
        )
        if bbox_diag < 5:
            return False

        n_verts = len(vertices)
        check_pts = pts[::5]  # Check every 5th point for speed
        max_dev = 0.0

        for pt in check_pts:
            min_dist = float("inf")
            for j in range(n_verts):
                a = vertices[j]
                b = vertices[(j + 1) % n_verts]

                # Point-to-line-segment distance
                ab = b - a
                ap = pt - a
                t = np.clip(
                    np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10), 0, 1
                )
                closest = a + t * ab
                dist = np.linalg.norm(pt - closest)
                min_dist = min(min_dist, dist)

            max_dev = max(max_dev, min_dist)

        deviation_pct = (max_dev / bbox_diag) * 100
        return deviation_pct < self.polygon_deviation_pct

    # ── Color Normalization ───────────────────────────────────────

    def _normalize_color(self, hex_color: str) -> str:
        """Normalize near-black/white hex colors."""
        if not hex_color or hex_color == "none":
            return hex_color

        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color

        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            return hex_color

        if r < 10 and g < 10 and b < 10:
            return "#000"
        if r > 250 and g > 250 and b > 250:
            return "#FFF"
        if abs(r - g) < 5 and abs(g - b) < 5:
            avg = (r + g + b) // 3
            return f"#{avg:02X}{avg:02X}{avg:02X}"

        return hex_color

    # ── SVG Parsing ───────────────────────────────────────────────

    def _extract_viewbox(self, svg_path: Path) -> str:
        """Extract viewBox from SVG file."""
        text = svg_path.read_text()
        match = re.search(r'viewBox="([^"]+)"', text)
        if match:
            return match.group(1)

        w_match = re.search(r'width="(\d+)"', text)
        h_match = re.search(r'height="(\d+)"', text)
        if w_match and h_match:
            return f"0 0 {w_match.group(1)} {h_match.group(1)}"

        return "0 0 1024 1024"

    # ── Complex Path Extraction (for LLM handoff) ────────────────

    def get_complex_paths(self, svg_path: Path | str) -> list[dict]:
        """
        Get paths that weren't converted — for optional LLM processing.

        Returns list of dicts with index, fill, d attribute, bbox, segments
        for each complex path. Feed these to the LLM one at a time.
        """
        paths, attrs = svg2paths(str(svg_path))
        complex_paths = []

        for i, (path, attr) in enumerate(zip(paths, attrs)):
            fill = attr.get("fill", "none")
            shape_type, _ = self._classify(path, fill)

            if shape_type == "keep":
                try:
                    xmin, xmax, ymin, ymax = path.bbox()
                    bbox = {
                        "x": round(xmin),
                        "y": round(ymin),
                        "w": round(xmax - xmin),
                        "h": round(ymax - ymin),
                    }
                except Exception:
                    bbox = {}

                complex_paths.append(
                    {
                        "index": i,
                        "fill": fill,
                        "d": attr.get("d", ""),
                        "bbox": bbox,
                        "segments": len(path),
                    }
                )

        return complex_paths
