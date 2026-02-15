"""
Lerp — Vectorizer (Stage 3a)
=============================
Converts raster PNGs to SVG using vtracer, then optimizes with SVGO.

This produces the "raw trace" — mathematically literal but bloated.
The Snap Engine (Stage 3b) then cleans it into geometric primitives.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass

import vtracer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    VTRACER_MODE, VTRACER_COLOR_PRECISION, VTRACER_FILTER_SPECKLE,
    VTRACER_CORNER_THRESHOLD, VTRACER_MAX_ITERATIONS, OUTPUT_DIR,
)


@dataclass
class TraceResult:
    """Result from tracing a raster to SVG."""
    input_path: Path
    raw_svg_path: Path
    optimized_svg_path: Path
    raw_size_bytes: int
    optimized_size_bytes: int
    svgo_reduction_pct: float


class Vectorizer:
    """
    Traces raster images to SVG using vtracer + SVGO.
    
    Usage:
        vec = Vectorizer()
        result = vec.trace("icon.png")
        # result.optimized_svg_path → clean SVG ready for snap engine
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (OUTPUT_DIR / "vectors")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_svgo()

    def _check_svgo(self):
        """Verify SVGO is installed."""
        try:
            result = subprocess.run(["svgo", "--version"], capture_output=True, text=True)
            self.svgo_available = result.returncode == 0
        except FileNotFoundError:
            self.svgo_available = False
            print("⚠️ SVGO not found. Install with: npm install -g svgo")

    def trace(self, input_path: Path | str, name: str = None) -> TraceResult:
        """
        Full trace pipeline: vtracer → SVGO.
        
        Args:
            input_path: Path to raster PNG
            name: Optional base name for output files
            
        Returns:
            TraceResult with paths to raw and optimized SVGs
        """
        input_path = Path(input_path)
        name = name or input_path.stem

        # Step 1: vtracer
        raw_svg_path = self.output_dir / f"{name}_raw.svg"
        self._run_vtracer(input_path, raw_svg_path)
        raw_size = raw_svg_path.stat().st_size

        # Step 2: SVGO optimization
        optimized_svg_path = self.output_dir / f"{name}_optimized.svg"
        if self.svgo_available:
            self._run_svgo(raw_svg_path, optimized_svg_path)
            opt_size = optimized_svg_path.stat().st_size
        else:
            # Copy raw as "optimized" if SVGO not available
            optimized_svg_path.write_text(raw_svg_path.read_text())
            opt_size = raw_size

        reduction = (1 - opt_size / raw_size) * 100 if raw_size > 0 else 0

        return TraceResult(
            input_path=input_path,
            raw_svg_path=raw_svg_path,
            optimized_svg_path=optimized_svg_path,
            raw_size_bytes=raw_size,
            optimized_size_bytes=opt_size,
            svgo_reduction_pct=round(reduction, 1),
        )

    def _run_vtracer(self, input_path: Path, output_path: Path):
        """Run vtracer on a raster image."""
        vtracer.convert_image_to_svg_py(
            image_path=str(input_path),
            out_path=str(output_path),
            colormode="color",
            hierarchical="stacked",
            mode=VTRACER_MODE,
            filter_speckle=VTRACER_FILTER_SPECKLE,
            color_precision=VTRACER_COLOR_PRECISION,
            corner_threshold=VTRACER_CORNER_THRESHOLD,
            max_iterations=VTRACER_MAX_ITERATIONS,
            length_threshold=4.0,
            splice_threshold=45,
            path_precision=3,
        )

    def _run_svgo(self, input_path: Path, output_path: Path):
        """Run SVGO to minify SVG."""
        result = subprocess.run(
            ["svgo", str(input_path), "-o", str(output_path), "--multipass"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"⚠️ SVGO warning: {result.stderr[:200]}")
            # Fall back to copying raw
            output_path.write_text(input_path.read_text())

    def trace_batch(self, input_paths: list[Path],
                    project_name: str = "logo") -> list[TraceResult]:
        """Trace multiple rasters."""
        results = []
        for i, path in enumerate(input_paths):
            name = f"{project_name}_{i+1}"
            print(f"  ✏️ Tracing {i+1}/{len(input_paths)}: {path.name}")
            result = self.trace(path, name=name)
            print(f"    Raw: {result.raw_size_bytes:,}B → Optimized: {result.optimized_size_bytes:,}B "
                  f"({result.svgo_reduction_pct}% reduction)")
            results.append(result)
        return results
