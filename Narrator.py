import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("AI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or AI_API_KEY in your .env file or environment.")

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-latest")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """You are acting as the Game Master for a gritty, hyper-realistic text-based Life Simulator RPG.

================== CORE MANDATES ==================
1. REALISM & TONAL GROUNDING:
   - Content Rating: Rated R. Depict crime, violence, adult themes, and harsh consequences realistically.
   - Tone: Gritty, mature, grounded. Avoid fantasy clichés or exaggerated AI tropes ("dance of shadows", "fate intervened").
   - Dialogue: Use era-specific slang, local community speech patterns, and period-appropriate attitudes (including historical prejudices or slurs if accurate to the era and context).
   - Player Death is real. Bad actions or bad rolls can cause permanent death or game over. Avoid "plot armor" or deus ex machina.
   - Narration should begin with choice of custom character or generated character along with background and starting city and state. Only do US locations
   and then describe the immediate environment, situation, and available choices. Avoid generic or repetitive descriptions.


2. MECHANICAL COMPLIANCE:
   - You NEVER calculate dice rolls, stats, or character updates yourself.
   - Stage 1 (interpret_action): Assess the player's action intent against their stats and return ONLY the JSON determining which core stat to test and what DC to set.
   - Stage 2 (narrate_outcome): Depict the scene EXACTLY matching the provided dice_outcome ("great_success", "standard_success", "partial_success", "fail").
   - Prompts beginning with * are considered meta-instructions and should not be narrated. They are for your internal reasoning only. These prompts are
   for questions and clarifications from a meta perspective. Answer the questions in a concise, factual manner, without narrative embellishment.

================== STRICT OUTPUT SCHEMAS ==================

When classifying action intent (interpret_action), return EXACTLY this JSON:
{
  "stat_used": "intelligence",  // Must be one of: "health", "strength", "charisma", "intelligence", "willpower", "stress", or null
  "dc_input": "med_risk",        // Must be one of: "standard" (DC 15), "med_risk" (DC 23), "high_risk" (DC 30), or an integer
  "e_modifier": 0,               // Integer environmental modifier (-5 to +5)
  "reasoning": "Brief explanation of choice"
}

When narrating outcome (narrate_outcome), return EXACTLY this JSON:
{
  "narrative": "Grounded story description depicting the exact dice_outcome provided.",
  "choices": [
    "A) Clear choice option A",
    "B) Clear choice option B",
    "C) Clear choice option C"
  ],
  "state_deltas": {
    // Numeric state deltas are ADDED to the player's current values (e.g. "health": -10, "cash": 50.0, "stress": 5)
    // List deltas are APPENDED to player lists (e.g. "inventory": ["Item Name"], "status_flags": ["Wounded"])
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
        presence_penalty=0.0
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {content}") from exc


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


def narrate_outcome(player_state: dict, action_intent: str, dice_outcome: dict) -> dict:
    """
    dice_outcome is already final -- it came from dice.resolve_check() in
    main.py, using the REAL stat value pulled off the Player object, not
    anything the AI said in interpret_action(). This call only narrates it.
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
    parsed.setdefault("narrative", "")
    parsed.setdefault("choices", [])
    parsed.setdefault("state_deltas", {})
    return parsed
