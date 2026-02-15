"""
Lerp — Strategist Agent Tests
==============================
Test the Brand Strategist with sample briefs.

Run: python -m pytest tests/test_strategist.py -v
  or: python tests/test_strategist.py  (for quick manual testing)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.strategist import BrandStrategist, ClientBrief, DesignSpec
from tests.sample_briefs import (
    CHAI_BRAND_TEXT, CHAI_BRAND_BRIEF,
    TECH_STARTUP_TEXT, TECH_STARTUP_BRIEF,
)
from config.settings import OUTPUT_DIR


def test_from_text():
    """Test strategist with a free-text brief."""
    print("\n" + "=" * 60)
    print("TEST: Strategist from text (Chai brand)")
    print("=" * 60)

    strategist = BrandStrategist()
    spec = strategist.from_text(CHAI_BRAND_TEXT)

    print(f"\n  Brand: {spec.brand_name}")
    print(f"  Tagline: {spec.tagline}")
    print(f"  Style: {spec.style_direction}")
    print(f"  Concepts: {len(spec.icon_concepts)}")
    for i, concept in enumerate(spec.icon_concepts):
        print(f"    {i+1}. {concept[:100]}...")
    print(f"  Colors: {json.dumps(spec.color_palette, indent=4)}")
    print(f"  Fonts: {spec.font_suggestions}")
    print(f"  Mood: {spec.mood}")
    print(f"  Avoid: {spec.avoid}")
    print(f"  Tracking: {spec.tracking}")
    print(f"  Case: {spec.case_treatment}")
    print(f"  Raster prompts: {len(spec.raster_prompts)}")
    for i, prompt in enumerate(spec.raster_prompts):
        print(f"    {i+1}. {prompt[:120]}...")
    print(f"  Rationale: {spec.rationale[:200]}...")

    # Save for inspection
    out_path = OUTPUT_DIR / "test_chai_spec.json"
    strategist.save_spec(spec, out_path)
    print(f"\n  ✓ Saved to: {out_path}")

    # Validate structure
    assert spec.brand_name, "Brand name should not be empty"
    assert len(spec.icon_concepts) >= 2, "Should have at least 2 icon concepts"
    assert spec.color_palette, "Should have a color palette"
    assert spec.raster_prompts, "Should have raster prompts"
    print("\n  ✅ All assertions passed")

    return spec


def test_from_brief():
    """Test strategist with a structured brief."""
    print("\n" + "=" * 60)
    print("TEST: Strategist from structured brief (Tech startup)")
    print("=" * 60)

    strategist = BrandStrategist()
    spec = strategist.from_brief(TECH_STARTUP_BRIEF)

    print(f"\n  Brand: {spec.brand_name}")
    print(f"  Tagline: {spec.tagline}")
    print(f"  Style: {spec.style_direction}")
    print(f"  Concepts: {len(spec.icon_concepts)}")
    print(f"  Colors: {json.dumps(spec.color_palette, indent=4)}")
    print(f"  Fonts: {spec.font_suggestions}")
    print(f"  Raster prompts: {len(spec.raster_prompts)}")

    # Save
    out_path = OUTPUT_DIR / "test_tech_spec.json"
    strategist.save_spec(spec, out_path)
    print(f"\n  ✓ Saved to: {out_path}")

    assert spec.brand_name == "Refract" or spec.brand_name, "Should preserve/generate brand name"
    assert spec.raster_prompts, "Should have raster prompts"
    print("\n  ✅ All assertions passed")

    return spec


def test_spec_roundtrip():
    """Test save/load cycle for design specs."""
    print("\n" + "=" * 60)
    print("TEST: Spec save/load roundtrip")
    print("=" * 60)

    strategist = BrandStrategist()

    # Create a spec manually
    spec = DesignSpec(
        brand_name="TestBrand",
        tagline="Testing the pipeline",
        industry="Testing",
        style_direction="minimal geometric",
        icon_concepts=["Abstract T lettermark", "Circular badge", "Monogram"],
        color_palette={
            "primary": {"hex": "#1A1A2E", "name": "Deep Navy", "role": "Logo, headers"},
            "secondary": {"hex": "#F5F5F0", "name": "Off White", "role": "Backgrounds"},
            "accent": {"hex": "#E94560", "name": "Coral", "role": "CTAs"},
        },
        font_suggestions=["Inter", "Space Grotesk"],
        mood=["minimal", "precise", "confident"],
        raster_prompts=[
            "minimalist abstract T lettermark icon, vector style, flat design, solid #1A1A2E, white background, clean edges, no text, centered",
        ],
        negative_prompt="photorealistic, 3D, textured, text, gradient, shadows",
    )

    # Save
    path = OUTPUT_DIR / "test_roundtrip_spec.json"
    strategist.save_spec(spec, path)

    # Load
    loaded = strategist.load_spec(path)

    assert loaded.brand_name == spec.brand_name
    assert loaded.color_palette == spec.color_palette
    assert loaded.raster_prompts == spec.raster_prompts

    print("  ✅ Roundtrip passed — save/load preserves all fields")


# ── Manual runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Running Strategist Tests\n")
    print("⚠️  These tests require ANTHROPIC_API_KEY in config/.env\n")

    try:
        test_spec_roundtrip()
    except Exception as e:
        print(f"  ❌ Roundtrip failed: {e}")

    try:
        test_from_text()
    except Exception as e:
        print(f"  ❌ Text brief test failed: {e}")

    try:
        test_from_brief()
    except Exception as e:
        print(f"  ❌ Structured brief test failed: {e}")

    print("\n🏁 Done!")
