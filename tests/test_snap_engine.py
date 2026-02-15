"""
Lerp — Snap Engine Test (with Deterministic Detector)
======================================================
Tests the new 3-pass snap engine:
  Pass 0: Deterministic shape detection (always runs, no API key needed)
  Pass 1: LLM cleanup (optional, needs ANTHROPIC_API_KEY)
  Pass 2: Visual refinement (optional, needs ANTHROPIC_API_KEY + cairosvg)

Usage:
    # Deterministic only (no API key needed):
    python tests/test_snap_engine.py

    # With LLM pass:
    python tests/test_snap_engine.py --llm

    # With LLM + visual pass:
    python tests/test_snap_engine.py --llm --visual
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR
from src.pipeline.snap_engine import SnapEngine


def test_snap_engine(llm_pass: bool = False, visual_pass: bool = False):
    # Setup
    test_input_dir = OUTPUT_DIR / "test_inputs"
    if not test_input_dir.exists():
        print(f"❌ Create {test_input_dir} and put test SVGs there first.")
        print("   Copy the Recraft SVGs into data/output/test_inputs/")
        return

    svg_files = sorted(test_input_dir.glob("*.svg"))
    if not svg_files:
        print(f"❌ No SVGs found in {test_input_dir}")
        return

    mode_str = "deterministic only"
    if llm_pass and visual_pass:
        mode_str = "deterministic + LLM + visual"
    elif llm_pass:
        mode_str = "deterministic + LLM"

    print("=" * 60)
    print(f"🧪 SNAP ENGINE TEST ({mode_str})")
    print(f"   Input: {len(svg_files)} SVGs")
    print("=" * 60)

    snap = SnapEngine()

    for svg_path in svg_files:
        print(f"\n{'─' * 40}")
        print(f"📐 Processing: {svg_path.name}")
        print(f"   Input: {svg_path.stat().st_size:,}B")

        result = snap.clean(
            svg_path,
            name=svg_path.stem,
            llm_pass=llm_pass,
            visual_pass=visual_pass,
            design_context="Chai tea brand logo icon, geometric minimalist style"
        )

        print(f"   Output: {result.cleaned_svg_path}")
        print(f"   Detector: {result.detector_converted} converted, "
              f"{result.detector_kept} kept original")
        print(f"   LLM used: {'Yes' if result.llm_pass_used else 'No'}")
        print(f"   Nodes: {result.input_nodes} → {result.output_nodes} "
              f"({result.node_reduction_pct}% reduction)")
        print(f"   Size: {result.input_size_bytes:,}B → {result.output_size_bytes:,}B "
              f"({result.size_reduction_pct}% reduction)")
        print(f"   Primitives used: {', '.join(result.primitives_used)}")

        # Quality checks
        if result.size_reduction_pct > 10:
            print("   ✅ Good size reduction")
        else:
            print("   ⚠️ Minimal size reduction")

        has_primitives = any(p in result.primitives_used
                           for p in ['circle', 'rect', 'polygon', 'ellipse'])
        if has_primitives:
            print("   ✅ Using geometric primitives")
        else:
            print("   ⚠️ No geometric primitives detected")

    print(f"\n{'=' * 60}")
    print(f"📊 DONE — Check {OUTPUT_DIR / 'cleaned'} for results")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                       help="Run LLM cleanup on complex paths (needs API key)")
    parser.add_argument("--visual", action="store_true",
                       help="Run visual refinement pass (needs API key + cairosvg)")
    args = parser.parse_args()

    test_snap_engine(llm_pass=args.llm, visual_pass=args.visual)
