"""
Lerp — Brand Strategist Agent
==============================
Takes a natural language brand brief and produces a structured JSON design spec
that drives the entire downstream pipeline (raster generation, vectorization,
typography, QA).

Two modes:
1. FORM mode — structured client brief (from the Vantage Point questionnaire)
2. CHAT mode — conversational intake (3-5 smart questions, then spec)

The agent reasons about: color psychology, industry conventions, distinctiveness
from competitors, appropriate style direction, and font pairing.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from anthropic import Anthropic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ANTHROPIC_API_KEY, STRATEGIST_MODEL, FONT_CATEGORIES


# ─── Data Models ──────────────────────────────────────────────────────

@dataclass
class ClientBrief:
    """Structured input from the client questionnaire."""
    # Business basics
    brand_name: Optional[str] = None          # May be empty if we're naming
    product_description: str = ""
    market_region: str = ""
    price_point: str = ""                     # budget | mid-market | premium | luxury
    sales_channels: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    differentiator: str = ""

    # Customer
    target_customer: dict = field(default_factory=dict)  # name, age, job, city, day_in_life
    problem_solved: str = ""
    current_alternatives: str = ""

    # Creative unlocks
    film_reference: str = ""
    film_reference_why: str = ""
    brand_personality: str = ""               # "person at a party" answer
    admired_brands: list[dict] = field(default_factory=list)  # [{brand, why}]
    never_feel_like: str = ""
    unexpected_tension: str = ""

    # Visual preferences
    design_direction: list[str] = field(default_factory=list)  # minimal, bold, etc.
    colors_love: str = ""
    colors_hate: str = ""
    typography_feel: str = ""
    imagery_style: str = ""
    reference_images: list[str] = field(default_factory=list)  # URLs or paths

    # Constraints
    must_include_words: list[str] = field(default_factory=list)
    must_exclude_words: list[str] = field(default_factory=list)
    domain_requirements: list[str] = field(default_factory=list)


@dataclass
class DesignSpec:
    """Structured output — the design spec that drives the pipeline."""
    brand_name: str = ""
    tagline: str = ""
    industry: str = ""
    style_direction: str = ""
    icon_concepts: list[str] = field(default_factory=list)       # 3 distinct directions
    color_palette: dict = field(default_factory=dict)             # primary, secondary, accent (hex)
    typography_direction: str = ""
    font_suggestions: list[str] = field(default_factory=list)     # 2-3 specific fonts
    mood: list[str] = field(default_factory=list)                 # 3-5 mood words
    avoid: list[str] = field(default_factory=list)                # anti-patterns
    tracking: str = ""                                             # letter-spacing guidance
    case_treatment: str = ""                                       # all-caps, mixed, lowercase
    lockup_preference: str = ""                                    # horizontal, stacked, icon-only
    raster_prompts: list[str] = field(default_factory=list)       # Ready-to-use Recraft prompts
    negative_prompt: str = ""                                      # Recraft negative prompt
    rationale: str = ""                                            # Why these choices


# ─── System Prompt ────────────────────────────────────────────────────

STRATEGIST_SYSTEM_PROMPT = """You are the Brand Strategist for Lerp, an AI logo generation pipeline.

Your job: Take a brand brief and produce a precise JSON design specification that will drive AI image generation (Recraft V3) and typography assembly downstream.

## What You Do

1. ANALYZE the brief — understand the brand's positioning, target market, competitive landscape
2. REASON about design choices — color psychology, industry conventions, distinctiveness
3. PRODUCE a structured design spec with:
   - 3 distinct icon concept directions (described in words, not images)
   - A color palette (primary, secondary, accent with hex codes)
   - Typography direction + specific font suggestions
   - Ready-to-use image generation prompts for Recraft V3
   - Clear anti-patterns to avoid

## Design Intelligence

Apply these principles:
- **Color psychology:** Blue = trust, Green = growth/nature, Black = luxury, Orange = energy
- **Premium signals:** Wide letter-spacing, minimal marks, limited color palette (2-3 colors)
- **Industry awareness:** Tech favors geometric sans, luxury favors thin serifs, food favors warm tones
- **Distinctiveness:** Check what competitors do, then deliberately diverge
- **Tension:** The best brands hold a paradox (e.g., "heritage product, modern presentation")

## Recraft V3 Prompt Rules

Your raster prompts MUST follow this format for best results:
- Start with: "minimalist [concept] icon"
- Add style: "vector style, flat design, solid colors"
- Specify colors using hex: "solid #2D4A3E"
- Always include: "white background, clean edges, no text, centered"
- Keep prompts under 100 words

Your negative prompt should always include:
"photorealistic, 3D, textured, text, letters, words, gradient, complex, detailed, shadows, bevels, glossy"

## Available Fonts

You may suggest fonts from these curated categories:
""" + json.dumps(FONT_CATEGORIES, indent=2) + """

## Output Format

Respond with ONLY a JSON object matching this schema — no markdown, no explanation outside the JSON:

{
  "brand_name": "string",
  "tagline": "string (short, memorable)",
  "industry": "string",
  "style_direction": "string (e.g., 'minimal Scandinavian meets Indian warmth')",
  "icon_concepts": [
    "concept 1 description (2-3 sentences)",
    "concept 2 description",
    "concept 3 description"
  ],
  "color_palette": {
    "primary": {"hex": "#XXXXXX", "name": "string", "role": "string"},
    "secondary": {"hex": "#XXXXXX", "name": "string", "role": "string"},
    "accent": {"hex": "#XXXXXX", "name": "string", "role": "string"}
  },
  "typography_direction": "string (e.g., 'clean geometric sans, medium weight, generous tracking')",
  "font_suggestions": ["Font Name 1", "Font Name 2"],
  "mood": ["word1", "word2", "word3", "word4"],
  "avoid": ["anti-pattern 1", "anti-pattern 2"],
  "tracking": "string (e.g., '0.1em' or 'tight')",
  "case_treatment": "all-caps | mixed-case | lowercase",
  "lockup_preference": "horizontal | stacked | flexible",
  "raster_prompts": [
    "prompt for concept 1 (ready for Recraft V3)",
    "prompt for concept 2",
    "prompt for concept 3"
  ],
  "negative_prompt": "string",
  "rationale": "string (2-3 sentences explaining the strategic thinking)"
}
"""


# ─── Agent ────────────────────────────────────────────────────────────

class BrandStrategist:
    """
    Brand Strategist Agent — converts brief to design spec.
    
    Usage:
        strategist = BrandStrategist()
        
        # From structured brief
        spec = strategist.from_brief(client_brief)
        
        # From natural language
        spec = strategist.from_text("I'm starting a premium dog food brand called Timber...")
        
        # Conversational mode
        strategist.start_conversation()
        response = strategist.chat("I'm building a chai brand")
        # ... more messages ...
        spec = strategist.finalize()
    """

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        self.model = STRATEGIST_MODEL
        self.conversation_history: list[dict] = []
        self._load_anti_patterns()

    def _load_anti_patterns(self):
        """Load seed anti-patterns to inject into context."""
        anti_patterns_path = Path(__file__).parent.parent.parent / "config" / "anti_patterns.json"
        try:
            with open(anti_patterns_path) as f:
                self.anti_patterns = json.load(f)
        except FileNotFoundError:
            self.anti_patterns = {}

    def _build_brief_message(self, brief: ClientBrief) -> str:
        """Convert structured brief to a natural language message for the LLM."""
        parts = []

        if brief.brand_name:
            parts.append(f"Brand name: {brief.brand_name}")
        if brief.product_description:
            parts.append(f"Product: {brief.product_description}")
        if brief.market_region:
            parts.append(f"Market: {brief.market_region}")
        if brief.price_point:
            parts.append(f"Price point: {brief.price_point}")
        if brief.sales_channels:
            parts.append(f"Sales channels: {', '.join(brief.sales_channels)}")
        if brief.competitors:
            parts.append(f"Competitors: {', '.join(brief.competitors)}")
        if brief.differentiator:
            parts.append(f"Differentiator: {brief.differentiator}")

        if brief.target_customer:
            tc = brief.target_customer
            parts.append(f"Target customer: {tc.get('name', 'N/A')}, {tc.get('age', 'N/A')}, "
                        f"{tc.get('job', 'N/A')}, {tc.get('city', 'N/A')}")
            if tc.get('day_in_life'):
                parts.append(f"Their typical day: {tc['day_in_life']}")

        if brief.problem_solved:
            parts.append(f"Problem solved: {brief.problem_solved}")

        if brief.film_reference:
            parts.append(f"Film reference: {brief.film_reference} — {brief.film_reference_why}")
        if brief.brand_personality:
            parts.append(f"Brand personality (at a party): {brief.brand_personality}")
        if brief.admired_brands:
            for ab in brief.admired_brands:
                parts.append(f"Admired brand: {ab.get('brand', '')} — {ab.get('why', '')}")
        if brief.never_feel_like:
            parts.append(f"Must NEVER feel like: {brief.never_feel_like}")
        if brief.unexpected_tension:
            parts.append(f"Unexpected tension/paradox: {brief.unexpected_tension}")

        if brief.design_direction:
            parts.append(f"Design direction preference: {', '.join(brief.design_direction)}")
        if brief.colors_love:
            parts.append(f"Colors they love: {brief.colors_love}")
        if brief.colors_hate:
            parts.append(f"Colors they hate: {brief.colors_hate}")
        if brief.typography_feel:
            parts.append(f"Typography feel: {brief.typography_feel}")

        if brief.must_include_words:
            parts.append(f"Must include words: {', '.join(brief.must_include_words)}")
        if brief.must_exclude_words:
            parts.append(f"Must exclude words: {', '.join(brief.must_exclude_words)}")

        # Inject anti-patterns
        if self.anti_patterns:
            parts.append(f"\n--- ANTI-PATTERNS (must avoid) ---")
            if self.anti_patterns.get("logo_cliches"):
                parts.append(f"Logo clichés to avoid: {', '.join(self.anti_patterns['logo_cliches'])}")
            if self.anti_patterns.get("naming_patterns_overused"):
                parts.append(f"Naming patterns to avoid: {', '.join(self.anti_patterns['naming_patterns_overused'])}")

        return "\n".join(parts)

    def _parse_spec(self, response_text: str) -> DesignSpec:
        """Parse LLM JSON response into a DesignSpec."""
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # Remove first line
            if text.endswith("```"):
                text = text[:-3]
            elif "```" in text:
                text = text[:text.rfind("```")]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse strategist response as JSON: {e}\n\nRaw: {text[:500]}")

        spec = DesignSpec(
            brand_name=data.get("brand_name", ""),
            tagline=data.get("tagline", ""),
            industry=data.get("industry", ""),
            style_direction=data.get("style_direction", ""),
            icon_concepts=data.get("icon_concepts", []),
            color_palette=data.get("color_palette", {}),
            typography_direction=data.get("typography_direction", ""),
            font_suggestions=data.get("font_suggestions", []),
            mood=data.get("mood", []),
            avoid=data.get("avoid", []),
            tracking=data.get("tracking", "0.05em"),
            case_treatment=data.get("case_treatment", "mixed-case"),
            lockup_preference=data.get("lockup_preference", "horizontal"),
            raster_prompts=data.get("raster_prompts", []),
            negative_prompt=data.get("negative_prompt", ""),
            rationale=data.get("rationale", ""),
        )
        return spec

    def from_brief(self, brief: ClientBrief) -> DesignSpec:
        """
        Generate design spec from a structured client brief.
        Single-shot — no conversation.
        """
        message = self._build_brief_message(brief)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=STRATEGIST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )

        return self._parse_spec(response.content[0].text)

    def from_text(self, text: str) -> DesignSpec:
        """
        Generate design spec from a free-form text description.
        Single-shot — no conversation.
        """
        # Inject anti-patterns
        anti_pattern_note = ""
        if self.anti_patterns.get("logo_cliches"):
            anti_pattern_note = (
                f"\n\n--- ANTI-PATTERNS (must avoid in your design spec) ---\n"
                f"Logo clichés: {', '.join(self.anti_patterns['logo_cliches'])}\n"
                f"Naming clichés: {', '.join(self.anti_patterns.get('naming_patterns_overused', []))}"
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=STRATEGIST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text + anti_pattern_note}],
        )

        return self._parse_spec(response.content[0].text)

    # ── Conversational Mode ───────────────────────────────────────────

    CONVERSATION_SYSTEM = """You are the Brand Strategist for Lerp, an AI logo generation tool.

You're having a brief intake conversation with a client. Your goal is to gather enough
information to produce a design specification in 3-5 questions. Don't fatigue the user.

Ask smart, specific questions — not generic ones. Listen to what they say and build on it.

When you have enough information (usually after 3-5 exchanges), say "READY_TO_GENERATE"
followed by a summary of what you've gathered. The system will then ask you to produce
the final design spec.

Be warm, confident, and concise. You're a creative director, not a form."""

    def start_conversation(self):
        """Start a conversational intake session."""
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """
        Send a message in conversation mode.
        Returns the agent's response (either a question or READY_TO_GENERATE).
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self.CONVERSATION_SYSTEM,
            messages=self.conversation_history,
        )

        assistant_msg = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def finalize(self) -> DesignSpec:
        """
        After conversation, generate the final design spec.
        Call this after the agent says READY_TO_GENERATE.
        """
        # Add the generation request
        self.conversation_history.append({
            "role": "user",
            "content": "Generate the final design specification as JSON now."
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=STRATEGIST_SYSTEM_PROMPT,
            messages=self.conversation_history,
        )

        return self._parse_spec(response.content[0].text)

    def spec_to_dict(self, spec: DesignSpec) -> dict:
        """Convert DesignSpec to a serializable dict."""
        return asdict(spec)

    def save_spec(self, spec: DesignSpec, path: str | Path):
        """Save design spec to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.spec_to_dict(spec), f, indent=2)

    def load_spec(self, path: str | Path) -> DesignSpec:
        """Load design spec from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return DesignSpec(**data)
