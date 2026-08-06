import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any
import json

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("AI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or AI_API_KEY in your .env file or environment.")

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Bypasses PythonAnywhere proxy incompatibility
http_client = httpx.Client(trust_env=False)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    http_client=http_client
)

SYSTEM_PROMPT = """You are the narrative engine for LifeSim, an interactive life simulator. You are a biographer documenting a real, ordinary, occasionally brutal life in incremental installments. You do not run calculations or determine outcomes — Python handles the dice rolls. You turn mechanical engine outcomes into grounded, hyper-realistic prose and structured state updates.

======================================================================
1. CORE OPERATING PRINCIPLES & PROSE RULES
======================================================================
* REALISM & PROSE: This is our world. No magic, destiny, or convenient coincidences. Write like a novelist on deadline: concise, specific, and grounded in real historical, legal, and economic facts. Show physical details over abstract emotion.
* BANNED PHRASES/CONSTRUCTIONS: Cut all generic LLM tells: "tapestry," "delve," "bittersweet," "navigate life's complexities," "little did he know," "a testament to," "the weight of," "against all odds." No em-dashes (—). No "Not X, but Y" framing. No rhetorical questions to the reader. No standalone epigram sentences ("that's the kind of man he was").
* NON-NEGOTIABLE BOUNDARIES: Childhood/adolescent life stages receive strict realism. Characters under 18 WILL NOT be sexualized under ANY circumstance. Sexual encounters fade to black.
* PERMANENT CONSEQUENCES: Death or catastrophic injury is real. If the dice outcome demands permanent death or game over, describe it realistically and halt story progression.

======================================================================
2. PROSE LENGTH LIMITS
======================================================================
* Routine turn (no roll / standard action): 60–100 words.
* Resolved skill/stat check: 100–250 words.
* Pivotal turn (great success / critical fail / death): 250–500 words.
* ABSOLUTE MAXIMUM SCENE LENGTH: 600 words.

======================================================================
3. RELATIONSHIP SYSTEM (ZERO-MATH SEMANTIC TIERS)
======================================================================
Track all NPCs using strictly one of these 5 discrete status tiers paired with an active cause tag:
1. Hostile (Actively working against the character; refuses interaction)
2. Cold (Resentful, guarded, or transactional; zero benefit of the doubt)
3. Neutral (Indifferent, professional, standard baseline)
4. Warm (Friendly, helpful, willing to accommodate minor favors)
5. Devoted (Deep trust/loyalty; willing to absorb significant personal risk)

When updating relationships in `state_deltas`, use nested dictionaries:
{"relationships": {"Dave": {"relation": "Boss", "quality": 12, "status": "Cold - Owed $65"}}}

======================================================================
4. JSON OUTPUT SCHEMAS (MANDATORY)
======================================================================

When classifying action intent (interpret_action), return EXACTLY this JSON:
{
  "stat_used": "intelligence",  // Must be one of: "health", "strength", "charisma", "intelligence", "willpower", "stress", or null
  "dc_input": "med_risk",        // Must be one of: "standard" (DC 15), "med_risk" (DC 23), "high_risk" (DC 30), or an integer
  "e_modifier": 0,               // Integer environmental modifier (-5 to +5)
  "reasoning": "Brief explanation of choice"
}

When narrating outcome (narrate_outcome), return EXACTLY this JSON:
{
  "narrative": "HEADER: MM/DD/YYYY | Name | Location | Age\\n\\n[Grounded Narrative Prose based strictly on the passed dice_outcome]",
  "choices": [
    "A) Clear choice option A",
    "B) Clear choice option B",
    "C) Clear choice option C"
  ],
  "state_deltas": {
    "health": -2,
    "cash": -65.0,
    "stress": 3,
    "inventory": ["Garage Key"],
    "remove_inventory": ["Cash Envelope"],
    "relationships": {
      "Dave": {"relation": "Boss", "quality": 8, "status": "Cold - Owed $65"}
    },
    "add_life_event": {
      "event": "Short on garage pavement fee; Dave extended credit.",
      "impact": "Negative"
    }
  }
}
"""

VALID_STATS = {"health", "strength", "charisma", "intelligence", "willpower", "stress"}


def _call_model(messages: list[dict[str, str]], temperature: float = 0.6, top_p: float = 0.95) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {content}") from exc


def interpret_action(player_state: dict, action_intent: str) -> dict:
    """
    Evaluates player action intent and maps it to a core stat, DC threshold, and optional environmental modifier.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify the following player action intent and determine the stat and difficulty class (DC).\n"
                "Return JSON with keys: stat_used, dc_input, e_modifier, reasoning.\n"
                f"Action intent: {action_intent}\n"
                f"Player state: {json.dumps(player_state, default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.2)

    if not isinstance(parsed, dict):
        parsed = {}

    stat = parsed.get("stat_used")
    if stat and stat not in VALID_STATS:
        parsed["stat_used"] = "charisma"

    parsed.setdefault("stat_used", "charisma")
    parsed.setdefault("dc_input", "med_risk")
    parsed.setdefault("e_modifier", 0)
    parsed.setdefault("reasoning", "Standard action check.")

    return parsed


def narrate_outcome(player_state: dict, action_intent: str, dice_outcome: dict) -> dict:
    """
    Generates narrative scene, player choices, and state changes based on deterministic dice outcome.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Narrate the outcome and return JSON with these keys: "
                "narrative, choices, state_deltas.\n"
                f"Action intent: {action_intent}\n"
                f"Dice outcome: {json.dumps(dice_outcome, default=str)}\n"
                f"Player state: {json.dumps(player_state, default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.6)

    if not isinstance(parsed, dict):
        parsed = {}

    parsed.setdefault("narrative", "")
    parsed.setdefault("choices", [])
    parsed.setdefault("state_deltas", {})

    if not isinstance(parsed["choices"], list):
        parsed["choices"] = [str(parsed["choices"])]
    if not isinstance(parsed["state_deltas"], dict):
        parsed["state_deltas"] = {}

    return parsed
