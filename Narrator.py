from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """
[the ruleset text you already built with Gemini goes here]
"""

def interpret_action(player_state: dict, action_intent: str) -> dict:
    """
    Give the AI the player's free-text action plus their current state.
    It names WHICH stat and WHICH difficulty tier this tests -- it does
    not roll dice or invent a stat value, only classifies. You decide
    the exact keys, but something like:
        {"stat_used": "charisma", "dc_input": "med_risk", "e_modifier": 0, "reasoning": "..."}
    """
    # TODO: build the messages list (system role + a user turn describing
    #       action_intent and the relevant parts of player_state)
    # TODO: call client.chat.completions.create(..., response_format=...)
    # TODO: json.loads() the content and return it
    # TODO: before trusting stat_used, check it's a real field on Player --
    #       the AI can still hallucinate a stat name that doesn't exist
    pass


def narrate_outcome(player_state: dict, action_intent: str, dice_outcome: dict) -> dict:
    """
    dice_outcome is already final -- it came from dice.resolve_check() in
    main.py, using the REAL stat value pulled off the Player object, not
    anything the AI said in interpret_action(). This call only narrates it.
    You decide the exact keys, but something like:
        {"narrative": "...", "choices": [...], "state_deltas": {...}}
    """
    # TODO: build messages (system role + user turn with action_intent,
    #       dice_outcome["outcome"], and enough of player_state for context)
    # TODO: call client.chat.completions.create()
    # TODO: parse and return
    pass