"""
Lerp — Snap Engine Manual Test
================================
Run this in VS Code with your ANTHROPIC_API_KEY set in config/.env

This takes the raw Recraft SVGs and runs them through the actual
Snap Engine (Claude API) to see if the automated cleanup matches
the manual cleanup.

Usage:
    python tests/test_snap_engine.py

Place the 3 test SVGs in data/output/test_inputs/ first:
    - minimalist-geometric-chai-glass-icon--tapered-trap.svg
    - minimalist-overlapping-diamond-shapes-icon--2-3-la__1_.svg
    - minimalist-overlapping-diamond-shapes-icon--2-3-la.svg
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR
from src.pipeline.snap_engine import SnapEngine


def test_snap_engine():
    # Setup
    test_input_dir = OUTPUT_DIR / "test_inputs"
    if not test_input_dir.exists():
        print(f"❌ Create {test_input_dir} and put the 3 test SVGs there first.")
        print("   Copy the Recraft SVGs into data/output/test_inputs/")
        return

    svg_files = sorted(test_input_dir.glob("*.svg"))
    if not svg_files:
        print(f"❌ No SVGs found in {test_input_dir}")
        return

    print("=" * 60)
    print("🧪 SNAP ENGINE TEST")
    print(f"   Input: {len(svg_files)} SVGs")
    print("=" * 60)

    snap = SnapEngine()

    for svg_path in svg_files:
        print(f"\n{'─' * 40}")
        print(f"📐 Processing: {svg_path.name}")
        print(f"   Input: {svg_path.stat().st_size:,}B")

        # Run Pass 1 only (no visual pass — faster, cheaper)
        result = snap.clean(
            svg_path,
            name=svg_path.stem,
            visual_pass=False,
            design_context="Chai tea brand logo icon, geometric minimalist style"
        )

        print(f"   Output: {result.cleaned_svg_path}")
        print(f"   Nodes: {result.input_nodes} → {result.output_nodes} "
              f"({result.node_reduction_pct}% reduction)")
        print(f"   Size: {result.input_size_bytes:,}B → {result.output_size_bytes:,}B "
              f"({result.size_reduction_pct}% reduction)")
        print(f"   Primitives used: {', '.join(result.primitives_used)}")

        # Check quality
        if result.node_reduction_pct > 50:
            print("   ✅ Good reduction")
        else:
            print("   ⚠️ Low reduction — may need better prompting")

        if 'circle' in result.primitives_used or 'rect' in result.primitives_used or 'polygon' in result.primitives_used:
            print("   ✅ Using geometric primitives")
        else:
            print("   ⚠️ Still using only <path> — snap engine didn't convert to primitives")

    print(f"\n{'=' * 60}")
    print("📊 DONE — Check data/output/cleaned/ for results")
    print("   Compare against the manual cleanups to see quality difference")
    print(f"{'=' * 60}")


def test_snap_with_visual_pass():
    """Same test but with Pass 2 (vision model). Costs more but better results."""
    test_input_dir = OUTPUT_DIR / "test_inputs"
    svg_files = sorted(test_input_dir.glob("*.svg"))

    if not svg_files:
        print("❌ No SVGs found")
        return

    snap = SnapEngine()

    # Just test one file with visual pass
    svg_path = svg_files[0]
    print(f"\n🔬 Visual pass test: {svg_path.name}")

    result = snap.clean(
        svg_path,
        name=f"{svg_path.stem}_visual",
        visual_pass=True,
        design_context="Chai tea brand logo icon, geometric minimalist style"
    )

    print(f"   Result: {result.cleaned_svg_path}")
    print(f"   Nodes: {result.input_nodes} → {result.output_nodes}")
    print(f"   Primitives: {', '.join(result.primitives_used)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Run with visual pass (Pass 2)")
    args = parser.parse_args()

    if args.visual:
        test_snap_with_visual_pass()
    else:
        test_snap_engine()
