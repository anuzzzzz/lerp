"""
Lerp — Snap Engine (Stage 3b)
==============================
The core technical differentiator.

Takes bloated vtracer SVG output and rewrites it using clean geometric
primitives (<circle>, <rect>, <polygon>) instead of opaque <path> data.

Two-pass approach:
  Pass 1: Rendered image + SVG code → faithful geometric simplification
  Pass 2: Render screenshot + SVG code → visual refinement
"""

import json
import base64
import subprocess
from pathlib import Path
from dataclasses import dataclass
from anthropic import Anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ANTHROPIC_API_KEY, SNAP_ENGINE_MODEL, OUTPUT_DIR


@dataclass
class SnapResult:
    """Result from the snap engine cleanup."""
    input_svg_path: Path
    cleaned_svg_path: Path
    input_nodes: int
    output_nodes: int
    input_size_bytes: int
    output_size_bytes: int
    node_reduction_pct: float
    size_reduction_pct: float
    primitives_used: list[str]   # e.g., ["circle", "rect", "polygon"]


# ─── Prompts ──────────────────────────────────────────────────────────

SNAP_PASS1_SYSTEM = """You are a precision SVG optimizer. Your ONLY job is to faithfully simplify SVG code while preserving the EXACT visual appearance.

## CRITICAL RULE: FAITHFUL REPRODUCTION

You must produce an SVG that looks IDENTICAL to the input when rendered. You are NOT a designer. You do NOT interpret, redesign, or improve the image. You SIMPLIFY the code while keeping the visual output the same.

NEVER:
- Add shapes that aren't in the original
- Remove shapes that are in the original
- Change the overall composition or layout
- Reinterpret what the image "should" look like
- Add elements like handles, saucers, steam that aren't there

ALWAYS:
- Preserve every visible shape's position, size, and color
- Keep the same viewBox dimensions
- Maintain all fill colors exactly (use the hex codes from the input)

## Simplification Rules (apply only where they preserve appearance)

1. **Closed paths with ~4 corners at ~90°** → `<rect>` or `<polygon>`
2. **Closed paths approximating circles** → `<circle>` or `<ellipse>`
3. **Closed paths made of straight segments** → `<polygon points="..."/>`
4. **Near-straight lines** → snap to exact horizontal/vertical
5. **Redundant anchor points on the same curve** → remove extras
6. **Coordinates** → round to whole numbers
7. **Near-identical colors** → normalize (#000002 → #000, #FEFEFE → #FFF)

## Process

1. Study the rendered image to understand what each shape IS
2. Read the SVG code to understand the structure
3. For each <path>, determine if it can become a simpler primitive
4. If a path is genuinely complex/organic, keep it as a simplified <path>
5. Verify your output would render identically to the input

## Output

- Valid SVG with proper xmlns
- XML comments describing each shape (e.g., <!-- Glass body -->)
- Output ONLY the SVG code — no markdown, no explanation"""


SNAP_PASS2_SYSTEM = """You are refining a cleaned SVG logo. You receive the cleaned SVG code and a rendered screenshot.

Compare the rendering against what a clean version should look like.

Fix ONLY:
- Shapes that are visibly wrong (misaligned, wrong size, wrong position)
- Missing shapes from the original
- Color mismatches
- Symmetry that should exist but is broken

Do NOT redesign or reinterpret. Output corrected SVG code only."""


class SnapEngine:
    """
    LLM-powered SVG cleanup engine.
    
    Usage:
        snap = SnapEngine()
        result = snap.clean("traced_logo.svg")
        # result.cleaned_svg_path → production-quality SVG
        
        # With visual refinement (requires cairosvg):
        result = snap.clean("traced_logo.svg", visual_pass=True)
    """

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self.model = SNAP_ENGINE_MODEL
        self.output_dir = OUTPUT_DIR / "cleaned"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, svg_path: Path | str, name: str = None,
              visual_pass: bool = False, design_context: str = "") -> SnapResult:
        """
        Run the snap engine on a traced SVG.
        
        Args:
            svg_path: Path to the vtracer + SVGO output
            name: Base name for output file
            visual_pass: Whether to do Pass 2 (visual refinement)
            design_context: Optional context about what the logo represents
            
        Returns:
            SnapResult with metrics comparing before/after
        """
        svg_path = Path(svg_path)
        name = name or svg_path.stem.replace("_optimized", "").replace("_raw", "")

        svg_code = svg_path.read_text()
        input_size = len(svg_code.encode())

        # ── Pass 1: Code-based cleanup ────────────────────────────────
        context_note = ""
        if design_context:
            context_note = f"\n\nContext about this logo: {design_context}"

        cleaned_code = self._pass1_code_cleanup(svg_code, context_note)

        # ── Pass 2: Visual refinement (optional) ──────────────────────
        if visual_pass:
            try:
                cleaned_code = self._pass2_visual_refinement(cleaned_code, name)
            except Exception as e:
                print(f"  ⚠️ Visual pass failed (using Pass 1 output): {e}")

        # ── Final SVGO pass ───────────────────────────────────────────
        output_path = self.output_dir / f"{name}_snapped.svg"
        output_path.write_text(cleaned_code)

        # Run SVGO on the cleaned output for final optimization
        final_path = self.output_dir / f"{name}_final.svg"
        try:
            subprocess.run(
                ["svgo", str(output_path), "-o", str(final_path), "--multipass"],
                capture_output=True, text=True,
            )
            if final_path.exists():
                cleaned_code = final_path.read_text()
                output_path = final_path
        except Exception:
            pass  # Skip if SVGO fails, use pre-SVGO output

        # ── Metrics ───────────────────────────────────────────────────
        output_size = len(cleaned_code.encode())
        input_nodes = self._count_nodes(svg_code)
        output_nodes = self._count_nodes(cleaned_code)

        return SnapResult(
            input_svg_path=svg_path,
            cleaned_svg_path=output_path,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            node_reduction_pct=round((1 - output_nodes / max(input_nodes, 1)) * 100, 1),
            size_reduction_pct=round((1 - output_size / max(input_size, 1)) * 100, 1),
            primitives_used=self._detect_primitives(cleaned_code),
        )

    def _pass1_code_cleanup(self, svg_code: str, context: str = "") -> str:
        """Pass 1: Image + code analysis — recognize shapes, snap to primitives."""
        import cairosvg

        # Render the input SVG to a screenshot
        has_image = False
        img_b64 = ""
        try:
            png_data = cairosvg.svg2png(
                bytestring=svg_code.encode(),
                output_width=512,
                output_height=512,
            )
            img_b64 = base64.b64encode(png_data).decode()
            has_image = True
        except Exception:
            pass

        # Build message with image + code
        content = []

        if has_image:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                }
            })

        content.append({
            "type": "text",
            "text": (
                "Above is the rendered SVG. Below is the SVG code.\n\n"
                "Simplify the SVG code using geometric primitives (polygon, rect, circle, ellipse) "
                "while producing output that looks IDENTICAL to the rendered image above.\n\n"
                "Do NOT redesign, reinterpret, or add/remove any shapes. "
                "Just simplify the code representation of what you see.\n\n"
                f"{svg_code}{context}"
            ),
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=SNAP_PASS1_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

        result = self._strip_code_fences(response.content[0].text)

        if not self._validate_svg(result):
            print("  ⚠️ Pass 1 produced invalid SVG, falling back to input")
            return svg_code

        return result

    def _pass2_visual_refinement(self, svg_code: str, name: str) -> str:
        """Pass 2: Vision + code — render screenshot, identify visual issues."""
        import cairosvg

        # Render the Pass 1 SVG to PNG
        render_path = self.output_dir / f"{name}_pass1_render.png"
        cairosvg.svg2png(
            bytestring=svg_code.encode(),
            write_to=str(render_path),
            output_width=512,
            output_height=512,
        )

        # Read and encode the rendered image
        with open(render_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Send both SVG code and rendered image to Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=SNAP_PASS2_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Here's the rendered SVG above, and the code below. "
                            "Fix any geometric issues you can see:\n\n"
                            f"{svg_code}"
                        ),
                    }
                ],
            }],
        )

        result = self._strip_code_fences(response.content[0].text)

        if not self._validate_svg(result):
            print("  ⚠️ Pass 2 produced invalid SVG, falling back to Pass 1")
            return svg_code

        return result

    def _strip_code_fences(self, text: str) -> str:
        """Strip markdown code fences from LLM output."""
        result = text.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1]
            if "```" in result:
                result = result[:result.rfind("```")]
            result = result.strip()
        return result

    def _validate_svg(self, svg_code: str) -> bool:
        """Check that output is minimally valid SVG."""
        return "<svg" in svg_code and "</svg>" in svg_code

    def _count_nodes(self, svg_code: str) -> int:
        """Count approximate node/element count in SVG."""
        count = 0
        # Count path commands
        import re
        # Count individual path d commands
        paths = re.findall(r'd="([^"]*)"', svg_code)
        for d in paths:
            # Count M, L, C, S, Q, A, Z commands (case insensitive)
            count += len(re.findall(r'[MLCSQAZmlcsqaz]', d))
        # Count primitive elements
        for elem in ['circle', 'rect', 'ellipse', 'line', 'polygon', 'polyline']:
            count += svg_code.count(f'<{elem}')
        return max(count, 1)

    def _detect_primitives(self, svg_code: str) -> list[str]:
        """Detect which SVG primitive types are used."""
        primitives = []
        for elem in ['circle', 'rect', 'ellipse', 'line', 'polygon', 'polyline', 'path']:
            if f'<{elem}' in svg_code:
                primitives.append(elem)
        return primitives

    def clean_batch(self, svg_paths: list[Path],
                    project_name: str = "logo",
                    visual_pass: bool = False) -> list[SnapResult]:
        """Clean multiple SVGs."""
        results = []
        for i, path in enumerate(svg_paths):
            name = f"{project_name}_{i+1}"
            print(f"  🧹 Cleaning {i+1}/{len(svg_paths)}: {path.name}")
            result = self.clean(path, name=name, visual_pass=visual_pass)
            print(f"    Nodes: {result.input_nodes} → {result.output_nodes} "
                  f"({result.node_reduction_pct}% reduction)")
            print(f"    Size: {result.input_size_bytes:,}B → {result.output_size_bytes:,}B "
                  f"({result.size_reduction_pct}% reduction)")
            print(f"    Primitives: {', '.join(result.primitives_used)}")
            results.append(result)
        return results
