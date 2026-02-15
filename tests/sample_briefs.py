"""
Lerp — Sample Briefs for Testing
=================================
These are structured ClientBrief objects and free-text descriptions
for testing the pipeline end-to-end.
"""

from src.agents.strategist import ClientBrief


# ─── Brief 1: Premium Chai Brand (from Vantage Point spec) ───────────

CHAI_BRAND_TEXT = """
I'm starting a premium single-origin chai tea brand targeting design-conscious 
millennials in Indian metros. Price point is premium — ₹800 per 100g tin.

Competitors are Vahdam Teas, Blue Tokai (tea line), and Chaayos retail.
My differentiator: we source single-estate Assam leaves and blend with 
whole spices — no powders, no shortcuts.

Sold D2C first, then premium retail.

My ideal customer is Priya, 28, UX designer at a Bangalore startup. She 
cares about design, buys Aesop soap and Blue Tokai coffee. She's tired of 
chai that looks like it belongs in her grandmother's kitchen.

Film reference: Grand Budapest Hotel — the precision, the color palette, 
the elevated mundane.

Brand personality at a party: The person who's quiet but everyone gravitates 
to. Dressed impeccably but not flashy. Makes one brilliant observation that 
shifts the whole conversation.

Admired brands: Aesop (ingredient storytelling), Muji (restraint), 
Paper Boat (nostalgia done right).

Must NEVER feel like: Generic "heritage Indian" or "spice market exotic".

The tension: Your grandmother's chai recipe, designed in Copenhagen.

Visual direction: Minimal, premium.
Colors I love: Deep greens, warm neutrals, copper.
Colors I hate: Terracotta, bright orange, gold.
Typography: Modern serif or clean sans.

Words that must NEVER appear: authentic, artisanal, handcrafted, curated.
Domain: .com or .co preferred.
"""

CHAI_BRAND_BRIEF = ClientBrief(
    brand_name=None,  # We want the system to help name it
    product_description="Premium single-origin chai tea, single-estate Assam leaves blended with whole spices",
    market_region="Indian metros (Bangalore, Mumbai, Delhi)",
    price_point="premium",
    sales_channels=["D2C", "Premium retail"],
    competitors=["Vahdam Teas", "Blue Tokai", "Chaayos"],
    differentiator="Single-estate sourcing, whole spices only, design-forward packaging",
    target_customer={
        "name": "Priya",
        "age": "28",
        "job": "UX Designer at Bangalore startup",
        "city": "Bangalore",
        "day_in_life": "Starts day with pour-over coffee, works from WeWork, evening chai ritual",
        "cares_about": "Design, sustainability, quality over quantity",
    },
    problem_solved="Chai that matches her taste in design — not kitsch, not basic",
    current_alternatives="Blue Tokai coffee, generic loose leaf from local store",
    film_reference="Grand Budapest Hotel",
    film_reference_why="The precision, color palette, elevated mundane, Wes Anderson symmetry",
    brand_personality="Quiet but magnetic. Impeccably dressed, not flashy. Makes one brilliant observation.",
    admired_brands=[
        {"brand": "Aesop", "why": "Ingredient storytelling, intellectual tone"},
        {"brand": "Muji", "why": "Restraint, lets quality speak"},
        {"brand": "Paper Boat", "why": "Nostalgia done right, not kitsch"},
    ],
    never_feel_like="Generic 'heritage Indian' or 'spice market exotic'",
    unexpected_tension="Your grandmother's chai recipe, designed in Copenhagen",
    design_direction=["Minimal", "Premium"],
    colors_love="Deep greens, warm neutrals, copper",
    colors_hate="Terracotta, bright orange, gold",
    typography_feel="Modern serif or clean sans-serif",
    must_exclude_words=["authentic", "artisanal", "handcrafted", "curated"],
    domain_requirements=[".com", ".co"],
)


# ─── Brief 2: Tech Startup (simple text brief) ───────────────────────

TECH_STARTUP_TEXT = """
I'm building an AI code review tool called Refract. It helps senior engineers 
review PRs 3x faster by surfacing the important changes and flagging patterns.

Target: Engineering leads at Series A-C startups. Think someone who uses Linear, 
loves good tooling, and is tired of reviewing 50 PRs a week.

Competitors: GitHub Copilot, CodeRabbit, Graphite.
Differentiator: We don't just find bugs — we understand architectural patterns 
and flag when code diverges from established patterns in your codebase.

Film reference: Her (2013) — warm, human, approachable tech.

Brand personality: The senior engineer who gives the best code reviews — thorough 
but kind, opinionated but open to discussion.

Visual direction: Modern, clean, slightly warm. Not cold enterprise SaaS.
Colors: Cool blues and warm accents. No purple (overused in dev tools).
Typography: Geometric sans-serif, monospace accents.
"""

TECH_STARTUP_BRIEF = ClientBrief(
    brand_name="Refract",
    product_description="AI code review tool that surfaces important changes and flags pattern divergence",
    market_region="Global (US primary)",
    price_point="premium",
    sales_channels=["D2C"],
    competitors=["GitHub Copilot", "CodeRabbit", "Graphite"],
    differentiator="Understands architectural patterns, not just bugs",
    target_customer={
        "name": "Marcus",
        "age": "32",
        "job": "Engineering Lead at Series B startup",
        "city": "San Francisco",
    },
    film_reference="Her (2013)",
    film_reference_why="Warm, human, approachable tech",
    brand_personality="Senior engineer who gives the best code reviews — thorough but kind",
    design_direction=["Modern", "Clean"],
    colors_love="Cool blues, warm accents",
    colors_hate="Purple",
    typography_feel="Geometric sans-serif with monospace accents",
)


# ─── Brief 3: Fitness Brand (minimal brief) ──────────────────────────

FITNESS_BRAND_TEXT = """
Premium home gym equipment brand called "Forge". Minimal, industrial aesthetic.
Target: Design-conscious fitness enthusiasts who want equipment that looks good 
in their living room. Think Peloton meets Vitsoe.

Competitors: Peloton, Technogym, Tonal.
Must feel: Strong, precise, built to last.
Must NOT feel: Bro gym culture, neon, aggressive.
Colors: Matte black, warm grey, brass accents.
"""


# ─── Brief 4: Restaurant Brand ───────────────────────────────────────

RESTAURANT_TEXT = """
Modern Indian restaurant in London called "Saffron Theory". Fine dining but 
not stuffy — think Dishoom energy but elevated. We deconstruct classic Indian 
dishes with modern techniques.

Target: London foodies, 25-40, who already love Dishoom and Gymkhana.
Film reference: Chef's Table on Netflix — that reverence for ingredients.
Must feel: Warm, intellectual, surprising.
Must NOT feel: Colonial nostalgia or "exotic East" orientalism.
"""


# ─── All briefs for batch testing ─────────────────────────────────────

ALL_TEXT_BRIEFS = {
    "chai": CHAI_BRAND_TEXT,
    "tech": TECH_STARTUP_TEXT,
    "fitness": FITNESS_BRAND_TEXT,
    "restaurant": RESTAURANT_TEXT,
}

ALL_STRUCTURED_BRIEFS = {
    "chai": CHAI_BRAND_BRIEF,
    "tech": TECH_STARTUP_BRIEF,
}
