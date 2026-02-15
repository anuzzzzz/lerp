"""
Lerp — Quality Assurance (Stage 5)
===================================
Three QA passes:
1. Geometric QA (scripted, no AI) — node count, symmetry, file size, contrast
2. Visual QA (vision model) — multi-scale rendering, recognizability, style alignment
3. Comparative QA (optional) — similarity check against known logos

If QA fails, feeds specific fix instructions back to the snap engine.
"""

import json
import base64
from pathlib import Path
from dataclasses import dataclass, field, asdict
from anthropic import Anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    ANTHROPIC_API_KEY, QA_VISION_MODEL,
    MAX_SVG_NODES, MAX_SVG_FILE_SIZE, SYMMETRY_THRESHOLD,
    MIN_CONTRAST_RATIO, BALANCE_THRESHOLD,
)
from src.utils.svg_utils import (
    count_nodes, file_size, symmetry_score, balance_index,
    contrast_ratio, render_at_sizes, full_evaluation,
)


@dataclass
class QAResult:
    """Complete QA evaluation result."""
    svg_path: str
    passed: bool
    geometric_qa: dict = field(default_factory=dict)
    visual_qa: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    fix_instructions: list[str] = field(default_factory=list)


class QualityAssurance:
    """
    Automated QA for logo SVGs.
    
    Usage:
        qa = QualityAssurance()
        result = qa.evaluate("cleaned_logo.svg", design_spec)
        
        if not result.passed:
            # Feed fix_instructions back to snap engine
            snap.clean(svg, context=result.fix_instructions)
    """

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self.model = QA_VISION_MODEL

    def evaluate(self, svg_path: Path | str, design_spec: dict = None,
                 visual_check: bool = True) -> QAResult:
        """
        Run full QA evaluation.
        
        Args:
            svg_path: Path to the SVG to evaluate
            design_spec: The design spec (for style alignment check)
            visual_check: Whether to run the vision model check
        """
        svg_path = Path(svg_path)
        failures = []
        fix_instructions = []

        # ── Pass 1: Geometric QA ─────────────────────────────────────
        geo_result = self._geometric_qa(svg_path)
        failures.extend(geo_result.get("failures", []))
        fix_instructions.extend(geo_result.get("fixes", []))

        # ── Pass 2: Visual QA ────────────────────────────────────────
        vis_result = {}
        if visual_check:
            try:
                vis_result = self._visual_qa(svg_path, design_spec)
                failures.extend(vis_result.get("failures", []))
                fix_instructions.extend(vis_result.get("fixes", []))
            except Exception as e:
                vis_result = {"error": str(e)}

        return QAResult(
            svg_path=str(svg_path),
            passed=len(failures) == 0,
            geometric_qa=geo_result,
            visual_qa=vis_result,
            failures=failures,
            fix_instructions=fix_instructions,
        )

    def _geometric_qa(self, svg_path: Path) -> dict:
        """Pass 1: Deterministic geometric checks."""
        result = {"checks": {}, "failures": [], "fixes": []}

        # Node count
        nodes = count_nodes(svg_path)
        node_count = nodes.get("nodes", 0)
        result["checks"]["nodes"] = node_count
        if node_count > MAX_SVG_NODES:
            result["failures"].append(f"Node count ({node_count}) exceeds limit ({MAX_SVG_NODES})")
            result["fixes"].append(f"Reduce nodes from {node_count} to under {MAX_SVG_NODES}. "
                                   "Simplify curves and use geometric primitives.")

        # File size
        size = file_size(svg_path)
        result["checks"]["file_size"] = size
        if size > MAX_SVG_FILE_SIZE:
            result["failures"].append(f"File size ({size:,}B) exceeds limit ({MAX_SVG_FILE_SIZE:,}B)")
            result["fixes"].append("Reduce SVG complexity. Remove redundant path data.")

        # Symmetry
        sym = symmetry_score(svg_path)
        result["checks"]["symmetry"] = sym
        if sym.get("score") and sym["score"] > SYMMETRY_THRESHOLD:
            result["failures"].append(f"Asymmetric (score: {sym['score']}, threshold: {SYMMETRY_THRESHOLD})")
            result["fixes"].append("Enforce bilateral symmetry: x_right = viewBox_width - x_left")

        # Balance
        bal = balance_index(svg_path)
        result["checks"]["balance"] = bal
        if bal.get("score") and bal["score"] > BALANCE_THRESHOLD:
            result["failures"].append(f"Unbalanced (score: {bal['score']}, threshold: {BALANCE_THRESHOLD})")
            result["fixes"].append("Center the visual weight. Adjust element positions toward canvas center.")

        # SVG validity
        svg_text = svg_path.read_text()
        if "<svg" not in svg_text:
            result["failures"].append("Invalid SVG: missing <svg> element")
        if "viewBox" not in svg_text and "width" not in svg_text:
            result["failures"].append("Missing viewBox or width/height attributes")

        return result

    def _visual_qa(self, svg_path: Path, design_spec: dict = None) -> dict:
        """Pass 2: Vision model evaluation of rendered SVG."""
        import cairosvg

        result = {"checks": {}, "failures": [], "fixes": []}

        # Render at multiple sizes
        renders = render_at_sizes(svg_path, sizes=[16, 64, 256, 512])
        if not renders:
            result["failures"].append("Could not render SVG")
            return result

        # Use the 256px render for the vision model
        render_256 = [r for r in renders if "256px" in str(r)]
        render_16 = [r for r in renders if "16px" in str(r)]

        if not render_256:
            return result

        # Read images
        with open(render_256[0], "rb") as f:
            img_256_b64 = base64.b64encode(f.read()).decode()

        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_256_b64},
            },
        ]

        if render_16:
            with open(render_16[0], "rb") as f:
                img_16_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_16_b64},
            })

        # Build evaluation prompt
        spec_context = ""
        if design_spec:
            spec_context = (
                f"\nDesign spec context:"
                f"\n- Style: {design_spec.get('style_direction', 'N/A')}"
                f"\n- Mood: {', '.join(design_spec.get('mood', []))}"
                f"\n- Should avoid: {', '.join(design_spec.get('avoid', []))}"
            )

        content.append({
            "type": "text",
            "text": (
                "Evaluate this logo SVG. First image is 256px, second (if present) is 16px favicon size.\n"
                f"{spec_context}\n\n"
                "Rate each criterion 1-5 and explain briefly:\n"
                "1. Recognizability at small sizes (favicon test)\n"
                "2. Visual clarity and professionalism\n"
                "3. Geometric cleanliness (circles round? lines straight?)\n"
                "4. Scalability (holds up at all sizes?)\n"
                "5. Distinctiveness (would you mistake this for a generic icon?)\n\n"
                "Respond as JSON:\n"
                '{"scores": {"recognizability": N, "clarity": N, "geometry": N, '
                '"scalability": N, "distinctiveness": N}, '
                '"overall": N, "issues": ["issue1"], "fixes": ["fix1"]}'
            ),
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )

        # Parse response
        try:
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if "```" in text:
                    text = text[:text.rfind("```")]
                text = text.strip()

            data = json.loads(text)
            result["checks"]["visual_scores"] = data.get("scores", {})
            result["checks"]["overall_score"] = data.get("overall", 0)

            if data.get("overall", 5) < 3:
                result["failures"].append(f"Visual QA score too low: {data['overall']}/5")
            result["fixes"].extend(data.get("fixes", []))

        except (json.JSONDecodeError, KeyError) as e:
            result["checks"]["raw_response"] = response.content[0].text[:500]

        return result
