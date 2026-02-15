"""
Lerp — Image Generator (Stage 2)
==================================
Two modes:
  PRIMARY:  Native SVG — Recraft generates vector directly ($0.08/image)
  FALLBACK: Raster PNG — then vtracer traces it ($0.04/image)

Native SVG still needs Snap Engine cleanup (excessive anchor points).
"""

import base64
import requests
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RECRAFT_API_KEY, RECRAFT_API_URL, RECRAFT_STYLE,
    RECRAFT_SIZE, RECRAFT_VARIANTS, OUTPUT_DIR,
)


class OutputFormat(Enum):
    SVG = "svg"
    RASTER = "png"


@dataclass
class GenerationResult:
    concept_index: int
    variant_index: int
    prompt: str
    output_path: Path
    output_format: OutputFormat
    file_size_bytes: int = 0
    needs_tracing: bool = False  # True=raster needs vtracer, False=already SVG


class ImageGenerator:
    """
    Usage:
        gen = ImageGenerator()
        results = gen.generate_from_spec(spec)              # auto: SVG first, raster fallback
        results = gen.generate_from_spec(spec, mode="svg")  # force SVG
        results = gen.generate_from_spec(spec, mode="png")  # force raster
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or RECRAFT_API_KEY
        self.svg_dir = OUTPUT_DIR / "vectors"
        self.raster_dir = OUTPUT_DIR / "rasters"
        self.svg_dir.mkdir(parents=True, exist_ok=True)
        self.raster_dir.mkdir(parents=True, exist_ok=True)

    def _call_recraft(self, prompt: str, negative_prompt: str = "",
                      style: str = None, n: int = 1,
                      fmt: OutputFormat = OutputFormat.SVG) -> list[dict]:
        if not self.api_key:
            raise ValueError("RECRAFT_API_KEY not set. Add it to config/.env")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        w, h = RECRAFT_SIZE.split("x")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or (
                "photorealistic, 3D, textured, text, letters, words, "
                "gradient, complex, detailed, shadows, bevels, glossy"
            ),
            "model": "recraftv3",
            "size": f"{w}x{h}",
            "n": n,
            "response_format": "b64_json",
        }

        if fmt == OutputFormat.SVG:
            payload["style"] = "vector_illustration"
            payload["substyle"] = "vector_illustration/flat_2"
        else:
            payload["style"] = style or RECRAFT_STYLE

        response = requests.post(
            RECRAFT_API_URL, headers=headers, json=payload, timeout=90,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Recraft API {response.status_code}: {response.text[:500]}")

        results = []
        for item in response.json().get("data", []):
            raw = base64.b64decode(item["b64_json"])
            is_svg = raw[:5] in (b"<?xml", b"<svg ", b"<svg\n")
            results.append({"data": raw, "is_svg": is_svg})

        return results

    def generate_from_spec(self, spec: dict, project_name: str = "logo",
                           mode: str = "auto") -> list[GenerationResult]:
        prompts = spec.get("raster_prompts", [])
        negative = spec.get("negative_prompt", "")

        if not prompts:
            raise ValueError("Design spec has no raster_prompts.")

        primary = OutputFormat.SVG if mode in ("auto", "svg") else OutputFormat.RASTER
        use_fallback = (mode == "auto")

        results = []

        for ci, prompt in enumerate(prompts):
            fmt = primary
            label = fmt.value.upper()
            print(f"  🎨 Concept {ci+1}/{len(prompts)}: {RECRAFT_VARIANTS} variants ({label})...")

            try:
                images = self._call_recraft(prompt, negative, n=RECRAFT_VARIANTS, fmt=fmt)
            except Exception as e:
                if use_fallback and fmt == OutputFormat.SVG:
                    print(f"    ⚠️ SVG failed: {e}\n    🔄 Falling back to raster...")
                    fmt = OutputFormat.RASTER
                    try:
                        images = self._call_recraft(prompt, negative, n=RECRAFT_VARIANTS, fmt=fmt)
                    except Exception as e2:
                        print(f"    ❌ Fallback failed: {e2}")
                        continue
                else:
                    print(f"    ❌ Failed: {e}")
                    continue

            for vi, img in enumerate(images):
                is_svg = img["is_svg"]
                ext = "svg" if is_svg else "png"
                out_dir = self.svg_dir if is_svg else self.raster_dir
                filename = f"{project_name}_c{ci+1}_v{vi+1}.{ext}"
                filepath = out_dir / filename

                with open(filepath, "wb") as f:
                    f.write(img["data"])

                results.append(GenerationResult(
                    concept_index=ci,
                    variant_index=vi,
                    prompt=prompt,
                    output_path=filepath,
                    output_format=OutputFormat.SVG if is_svg else OutputFormat.RASTER,
                    file_size_bytes=len(img["data"]),
                    needs_tracing=not is_svg,
                ))
                marker = "📐" if is_svg else "🖼️"
                print(f"    {marker} {filename} ({len(img['data']):,}B)")

        svg_n = sum(1 for r in results if not r.needs_tracing)
        png_n = sum(1 for r in results if r.needs_tracing)
        print(f"\n  ✓ {len(results)} total: {svg_n} SVG + {png_n} raster")
        return results
