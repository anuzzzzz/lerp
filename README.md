# Lerp

**Linear Interpolation from brief to brand mark.**

An AI-powered logo generation pipeline that takes a natural language brand brief and produces production-ready SVG logos. Named after `lerp` — the mathematical operation of smoothly transitioning between two points — because that's exactly what this does: smoothly interpolates from a rough idea to a clean vector mark.

## Architecture

```
Brand Brief (natural language)
    ↓
[1] Brand Strategist Agent — LLM intake → structured JSON design spec
    ↓
[2] Raster Generation — Recraft V3 API → flat icon PNGs
    ↓
[3] Snap Engine — vtracer → SVGO → Claude cleanup → clean SVG primitives
    ↓
[4] Typography Assembly — opentype.js → font-to-path + lockup composition
    ↓
[5] QA Loop — geometric checks + visual QA + multi-scale rendering
    ↓
Final Output: SVG logo + variations + brand kit
```

## Project Structure

```
lerp/
├── src/
│   ├── agents/
│   │   └── strategist.py      # Brand Strategist Agent (Claude)
│   ├── pipeline/
│   │   ├── raster.py           # Recraft V3 raster generation
│   │   ├── vectorizer.py       # vtracer + SVGO tracing
│   │   ├── snap_engine.py      # Claude SVG cleanup (the core IP)
│   │   ├── typography.py       # Font-to-path + lockup
│   │   └── qa.py               # Geometric + visual QA
│   └── utils/
│       ├── svg_utils.py        # SVG parsing, metrics, rendering
│       └── image_utils.py      # Color quantization, bg removal
├── config/
│   ├── settings.py             # API keys, model config
│   ├── fonts.json              # Curated font library metadata
│   └── anti_patterns.json      # Seed clichés to avoid
├── tests/
│   ├── test_strategist.py      # Test brand intake
│   ├── test_pipeline.py        # Test full pipeline
│   └── sample_briefs.py        # Test briefs (chai brand, etc.)
├── frontend/                   # React UI (Phase 2)
├── data/
│   ├── fonts/                  # Bundled .ttf/.otf files
│   ├── reference_library/      # Cached brand/film references
│   └── output/                 # Generated logos
├── requirements.txt
├── package.json                # For svgo + opentype.js
└── README.md
```

## Development Phases

| Phase | What | Status |
|-------|------|--------|
| Phase 1 | Brand Strategist Agent | 🔨 Now |
| Phase 2 | Raster Generation (Recraft V3) | Next |
| Phase 3 | Snap Engine (vtracer + Claude) | Next |
| Phase 4 | Typography Assembly | Later |
| Phase 5 | QA Loop | Later |
| Phase 6 | React Frontend | Later |

## Quick Start

```bash
# Install Python deps
pip install -r requirements.txt

# Install Node deps (for svgo)
npm install

# Set up API keys
cp config/.env.example config/.env
# Edit .env with your keys

# Run tests
python -m pytest tests/ -v

# Run full pipeline on a sample brief
python -m src.pipeline.run --brief tests/sample_briefs.py::chai_brand
```

## API Keys Required

- `ANTHROPIC_API_KEY` — Claude API for strategist + snap engine
- `RECRAFT_API_KEY` — Recraft V3 for raster generation
- `SERPAPI_KEY` — (optional) image search for visual research
