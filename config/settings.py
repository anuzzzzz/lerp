"""
Lerp — Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# ── API Keys ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
RECRAFT_API_KEY = os.getenv("RECRAFT_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ── Model Config ──────────────────────────────────────────────────────
STRATEGIST_MODEL = "claude-sonnet-4-20250514"   # Brand intake + design spec
SNAP_ENGINE_MODEL = "claude-sonnet-4-20250514"   # SVG cleanup (needs code reasoning)
QA_VISION_MODEL = "claude-sonnet-4-20250514"     # Visual QA (needs vision)

# ── Recraft V3 ────────────────────────────────────────────────────────
RECRAFT_API_URL = "https://external.api.recraft.ai/v1/images/generations"
RECRAFT_STYLE = "icon"              # "icon" or "vector_illustration"
RECRAFT_SIZE = "1024x1024"
RECRAFT_VARIANTS = 3                # Icons per concept direction
RECRAFT_CONCEPTS = 3                # Concept directions per brief

# ── Vectorizer ────────────────────────────────────────────────────────
VTRACER_MODE = "spline"             # "spline" for curves, "polygon" for straight
VTRACER_COLOR_PRECISION = 6
VTRACER_FILTER_SPECKLE = 4
VTRACER_CORNER_THRESHOLD = 60
VTRACER_MAX_ITERATIONS = 10

# ── Quality Thresholds ────────────────────────────────────────────────
MAX_SVG_NODES = 100                 # Target: < 100 for simple marks
MAX_SVG_FILE_SIZE = 5000            # Target: < 5KB
SYMMETRY_THRESHOLD = 0.15           # Hausdorff score, lower = better
MIN_CONTRAST_RATIO = 4.5            # WCAG AA minimum
BALANCE_THRESHOLD = 0.02            # Center of mass vs geometric center

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"
FONTS_DIR = DATA_DIR / "fonts"
REFERENCE_DIR = DATA_DIR / "reference_library"

# Ensure dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# ── Typography ────────────────────────────────────────────────────────
# Curated font categories for the strategist to pick from
FONT_CATEGORIES = {
    "geometric_sans": ["DM Sans", "Plus Jakarta Sans", "Outfit", "Space Grotesk"],
    "humanist_sans": ["Inter", "Source Sans 3", "Nunito Sans"],
    "modern_serif": ["GT Sectra", "Playfair Display", "Cormorant"],
    "slab_serif": ["Roboto Slab", "Zilla Slab"],
    "display": ["Bebas Neue", "Oswald", "Righteous"],
    "monospace": ["JetBrains Mono", "Space Mono", "IBM Plex Mono"],
}
