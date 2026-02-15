"""
Lerp — Full Pipeline Tests
============================
End-to-end tests that run the complete pipeline on sample briefs.

Run: python tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.run import LerpPipeline
from tests.sample_briefs import CHAI_BRAND_TEXT, TECH_STARTUP_TEXT, CHAI_BRAND_BRIEF


def test_full_pipeline_text():
    """Test full pipeline from text brief."""
    print("\n" + "=" * 60)
    print("TEST: Full pipeline — Chai brand (text brief)")
    print("=" * 60)

    pipeline = LerpPipeline(visual_qa=False, visual_snap=False)
    result = pipeline.run(CHAI_BRAND_TEXT, project_name="test_chai")

    print(f"\n  Stages completed: {list(result.get('stages', {}).keys())}")
    print(f"  Design spec: {'✓' if result.get('design_spec') else '✗'}")

    if "raster" in result.get("stages", {}):
        raster = result["stages"]["raster"]
        if "error" in raster:
            print(f"  Raster: ⚠️ {raster['error'][:100]}")
        else:
            print(f"  Rasters generated: {raster.get('count', 0)}")

    if "snap_engine" in result.get("stages", {}):
        snap = result["stages"]["snap_engine"]
        print(f"  SVGs cleaned: {snap.get('count', 0)}")
        print(f"  Avg node reduction: {snap.get('avg_node_reduction', 0):.1f}%")

    if "qa" in result.get("stages", {}):
        qa = result["stages"]["qa"]
        print(f"  QA: {qa.get('passed', 0)}/{qa.get('total', 0)} passed")

    return result


def test_full_pipeline_brief():
    """Test full pipeline from structured brief."""
    print("\n" + "=" * 60)
    print("TEST: Full pipeline — Chai brand (structured brief)")
    print("=" * 60)

    pipeline = LerpPipeline(visual_qa=False, visual_snap=False)
    result = pipeline.run_from_brief(CHAI_BRAND_BRIEF, project_name="test_chai_structured")

    return result


def test_strategist_only():
    """Test just the strategist stage (no API keys needed for Recraft)."""
    print("\n" + "=" * 60)
    print("TEST: Strategist only — Tech startup")
    print("=" * 60)

    from src.agents.strategist import BrandStrategist
    strategist = BrandStrategist()
    spec = strategist.from_text(TECH_STARTUP_TEXT)

    print(f"  Brand: {spec.brand_name}")
    print(f"  Concepts: {len(spec.icon_concepts)}")
    print(f"  Raster prompts: {len(spec.raster_prompts)}")

    for i, prompt in enumerate(spec.raster_prompts):
        print(f"  Prompt {i+1}: {prompt[:100]}...")

    print("\n  ✅ Strategist test passed")
    return spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["strategist", "full", "all"], default="strategist",
                       help="Which test to run")
    args = parser.parse_args()

    if args.stage == "strategist":
        test_strategist_only()
    elif args.stage == "full":
        test_full_pipeline_text()
    elif args.stage == "all":
        test_strategist_only()
        test_full_pipeline_text()
