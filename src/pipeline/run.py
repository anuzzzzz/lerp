"""
Lerp — Full Pipeline Runner (updated for hybrid SVG/raster)
=============================================================
Routes native SVGs directly to Snap Engine.
Routes raster PNGs through vtracer first, then Snap Engine.

Usage:
    python -m src.pipeline.run --brief "Premium dog food brand called Timber"
    python -m src.pipeline.run --brief "..." --mode svg    # force native SVG
    python -m src.pipeline.run --brief "..." --mode png    # force raster path
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import OUTPUT_DIR
from src.agents.strategist import BrandStrategist, ClientBrief, DesignSpec
from src.pipeline.generator import ImageGenerator, OutputFormat
from src.pipeline.vectorizer import Vectorizer
from src.pipeline.snap_engine import SnapEngine
from src.pipeline.qa import QualityAssurance
from src.utils.image_utils import prepare_for_tracing


class LerpPipeline:

    def __init__(self, visual_qa: bool = True, visual_snap: bool = True,
                 generation_mode: str = "auto"):
        self.strategist = BrandStrategist()
        self.generator = ImageGenerator()
        self.vectorizer = Vectorizer()
        self.snap_engine = SnapEngine()
        self.qa = QualityAssurance()
        self.visual_qa = visual_qa
        self.visual_snap = visual_snap
        self.generation_mode = generation_mode

    def run(self, brief_text: str, project_name: str = None) -> dict:
        project_name = project_name or f"lerp_{int(time.time())}"
        project_dir = OUTPUT_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"🎨 LERP — Logo Generation Pipeline")
        print(f"   Project: {project_name}")
        print(f"   Mode: {self.generation_mode}")
        print("=" * 60)

        # ── Stage 1: Brand Strategist ─────────────────────────────
        print(f"\n{'─' * 40}")
        print("📋 Stage 1: Brand Strategist")
        print(f"{'─' * 40}")

        spec = self.strategist.from_text(brief_text)
        spec_path = project_dir / "design_spec.json"
        self.strategist.save_spec(spec, spec_path)
        spec_dict = self.strategist.spec_to_dict(spec)

        print(f"  ✓ Brand: {spec.brand_name}")
        print(f"    Style: {spec.style_direction}")
        print(f"    Concepts: {len(spec.icon_concepts)}")

        return self._run_from_spec(spec_dict, project_name, project_dir)

    def run_from_brief(self, brief: ClientBrief, project_name: str = None) -> dict:
        project_name = project_name or f"lerp_{int(time.time())}"
        project_dir = OUTPUT_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        spec = self.strategist.from_brief(brief)
        self.strategist.save_spec(spec, project_dir / "design_spec.json")
        return self._run_from_spec(self.strategist.spec_to_dict(spec), project_name, project_dir)

    def run_from_spec(self, spec_path: str | Path, project_name: str = None) -> dict:
        spec = self.strategist.load_spec(spec_path)
        project_name = project_name or f"lerp_{int(time.time())}"
        project_dir = OUTPUT_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return self._run_from_spec(self.strategist.spec_to_dict(spec), project_name, project_dir)

    def _run_from_spec(self, spec_dict: dict, project_name: str,
                       project_dir: Path) -> dict:
        results = {
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "design_spec": spec_dict,
            "generation_mode": self.generation_mode,
            "stages": {},
        }

        # ── Stage 2: Generation ───────────────────────────────────
        print(f"\n{'─' * 40}")
        print("🖼️  Stage 2: Generation (Recraft V3)")
        print(f"{'─' * 40}")

        try:
            gen_results = self.generator.generate_from_spec(
                spec_dict, project_name, mode=self.generation_mode,
            )
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            results["stages"]["generation"] = {"error": str(e)}
            return results

        # Split into SVG (direct to snap) and raster (needs tracing)
        svg_results = [r for r in gen_results if not r.needs_tracing]
        raster_results = [r for r in gen_results if r.needs_tracing]

        results["stages"]["generation"] = {
            "total": len(gen_results),
            "native_svg": len(svg_results),
            "raster": len(raster_results),
        }

        # ── Stage 3a: Trace rasters (only if needed) ─────────────
        traced_svg_paths = []

        if raster_results:
            print(f"\n{'─' * 40}")
            print(f"✏️  Stage 3a: Tracing {len(raster_results)} rasters (vtracer + SVGO)")
            print(f"{'─' * 40}")

            # Extract palette for quantization
            palette_hex = []
            for color_data in spec_dict.get("color_palette", {}).values():
                if isinstance(color_data, dict) and "hex" in color_data:
                    palette_hex.append(color_data["hex"])
            palette_hex.append("#FFFFFF")

            for r in raster_results:
                try:
                    processed = prepare_for_tracing(r.output_path, palette_hex=palette_hex)
                    trace = self.vectorizer.trace(processed, name=r.output_path.stem)
                    traced_svg_paths.append(trace.optimized_svg_path)
                    print(f"    ✓ Traced: {r.output_path.name} → {trace.optimized_svg_path.name}")
                except Exception as e:
                    print(f"    ⚠️ Trace failed: {r.output_path.name}: {e}")

        # Collect all SVG paths for snap engine
        all_svg_paths = [r.output_path for r in svg_results] + traced_svg_paths

        if not all_svg_paths:
            print("  ❌ No SVGs to clean. Pipeline stopped.")
            return results

        # ── Stage 3b: Snap Engine ─────────────────────────────────
        print(f"\n{'─' * 40}")
        print(f"🧹 Stage 3b: Snap Engine ({len(all_svg_paths)} SVGs)")
        print(f"{'─' * 40}")

        snap_results = self.snap_engine.clean_batch(
            all_svg_paths,
            project_name=project_name,
            visual_pass=self.visual_snap,
        )

        results["stages"]["snap_engine"] = {
            "count": len(snap_results),
            "files": [str(sr.cleaned_svg_path) for sr in snap_results],
            "avg_node_reduction": (
                sum(sr.node_reduction_pct for sr in snap_results) / max(len(snap_results), 1)
            ),
        }

        # ── Stage 5: QA ──────────────────────────────────────────
        print(f"\n{'─' * 40}")
        print("✅ Stage 5: Quality Assurance")
        print(f"{'─' * 40}")

        qa_results = []
        for sr in snap_results:
            qr = self.qa.evaluate(
                sr.cleaned_svg_path,
                design_spec=spec_dict,
                visual_check=self.visual_qa,
            )
            qa_results.append(qr)
            status = "✅ PASS" if qr.passed else "❌ FAIL"
            print(f"  {status}: {sr.cleaned_svg_path.name}")
            if not qr.passed:
                for f in qr.failures:
                    print(f"    → {f}")

        passed = sum(1 for q in qa_results if q.passed)
        results["stages"]["qa"] = {
            "total": len(qa_results),
            "passed": passed,
            "failed": len(qa_results) - passed,
        }

        # ── Summary ───────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"📊 COMPLETE — QA passed: {passed}/{len(qa_results)}")
        print(f"   Output: {project_dir}")
        print(f"{'=' * 60}")

        results_path = project_dir / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results


def main():
    parser = argparse.ArgumentParser(description="Lerp — AI Logo Pipeline")
    parser.add_argument("--brief", type=str, help="Natural language brief")
    parser.add_argument("--spec", type=str, help="Path to saved design spec JSON")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--mode", choices=["auto", "svg", "png"], default="auto",
                        help="Generation mode: auto (SVG first, raster fallback), svg, png")
    parser.add_argument("--no-visual-qa", action="store_true")
    parser.add_argument("--no-visual-snap", action="store_true")
    args = parser.parse_args()

    pipeline = LerpPipeline(
        visual_qa=not args.no_visual_qa,
        visual_snap=not args.no_visual_snap,
        generation_mode=args.mode,
    )

    if args.brief:
        pipeline.run(args.brief, project_name=args.name)
    elif args.spec:
        pipeline.run_from_spec(args.spec, project_name=args.name)
    else:
        print("Provide --brief or --spec. Example:")
        print('  python -m src.pipeline.run --brief "Premium dog food brand called Timber"')
        print('  python -m src.pipeline.run --brief "..." --mode svg')


if __name__ == "__main__":
    main()
