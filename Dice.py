import random
from typing import Dict, Any
from Player import Player

#Convert 0-100 Stats into a dice roll modifier
def stat_to_modifier(stat):
    if stat < 0 or stat > 20:
        raise ValueError("Stat must be between 0 and 20")
    return (stat - 10) // 2

#Roll 2d20 and add the modifier
def roll_2d20(stat: int, e_modifier: int = 0):
    #Runs the stat_to_modifier function and assigns the return value to the variable 'modifier'.
    modifier = stat_to_modifier(stat)
    roll = random.randint(1, 20) + random.randint(1, 20)
    return roll + modifier + e_modifier

def _normalize_tier(raw: str) -> str:
    """Lowercase, strip, and collapse spaces/hyphens to underscores so trivial
    formatting differences ('Low Risk', 'low-risk', 'LOW_RISK') all match the
    same canonical key."""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


# Alternate phrasings the model has been observed to use (or could reasonably
# invent) that should resolve to a real tier instead of crashing the turn.
# Keys must already be normalized (see _normalize_tier).
TIER_ALIASES = {
    "low": "low_risk",
    "low_risk": "low_risk",
    "very_low_risk": "low_risk",
    "trivial": "low_risk",
    "easy": "low_risk",
    "routine": "low_risk",
    "standard": "standard",
    "normal": "standard",
    "medium": "med_risk",
    "moderate": "med_risk",
    "med_risk": "med_risk",
    "medium_risk": "med_risk",
    "high": "high_risk",
    "high_risk": "high_risk",
    "hard": "high_risk",
    "difficult": "high_risk",
    "very_high_risk": "high_risk",
    "extreme": "high_risk",
    # "auto" means: this action was never genuinely in doubt, don't roll at
    # all. Distinct from low_risk (which still rolls, just against an easy
    # DC) - auto skips the roll entirely. See NO_CHECK_TIER below.
    "auto": "auto",
    "automatic": "auto",
    "no_check": "auto",
    "no_roll": "auto",
    "none": "auto",
    "certain": "auto",
    "guaranteed": "auto",
}

# Sentinel tier: not a real numeric DC, means "skip the roll." Kept out of
# difficulty_classes on purpose so it can never be confused with a real DC
# value, and so any code that iterates difficulty_classes for real tiers
# doesn't accidentally pick it up.
NO_CHECK_TIER = "auto"


def resolve_dc_tier(dc_input) -> tuple:
    """
    Resolves a dc_input (string tier name or raw int) to an actual DC number.
    Returns (dc, resolved_tier_name). Never raises: an unrecognized string
    falls back to "standard" rather than crashing the turn, since a slightly
    wrong difficulty is a much smaller problem than the whole action failing
    with a roll of 0.

    For the no-check sentinel, dc is returned as None - callers should check
    is_no_check(dc_input) BEFORE calling resolve_check/resolve_dc_tier if
    they want to skip the roll instead of receiving a None dc.
    """
    if not isinstance(dc_input, str):
        return dc_input, "custom"

    normalized = _normalize_tier(dc_input)

    if normalized == NO_CHECK_TIER:
        return None, NO_CHECK_TIER

    if normalized in difficulty_classes:
        return difficulty_classes[normalized], normalized

    if normalized in TIER_ALIASES:
        resolved = TIER_ALIASES[normalized]
        if resolved == NO_CHECK_TIER:
            return None, NO_CHECK_TIER
        return difficulty_classes[resolved], resolved

    # Last resort: unrecognized tier entirely. Default to standard rather
    # than raising, and let the caller know a fallback occurred.
    return difficulty_classes["standard"], "standard (fallback)"


def is_no_check(dc_input) -> bool:
    """
    True if dc_input resolves to the no-check sentinel ("auto" or an alias
    of it). Callers (app.py, Main.py) should check this BEFORE calling
    resolve_check, and skip the roll entirely rather than calling
    resolve_check with a dc_input that will just come back as an
    auto-success anyway - this avoids a wasted call and keeps the
    "did we roll or not" decision explicit at the call site.
    """
    if not isinstance(dc_input, str):
        return False
    normalized = _normalize_tier(dc_input)
    if normalized == NO_CHECK_TIER:
        return True
    return TIER_ALIASES.get(normalized) == NO_CHECK_TIER


def auto_success_outcome() -> Dict[str, Any]:
    """
    The dice_outcome shape to use when an action bypasses the roll entirely
    because it was never genuinely in doubt (routine, low-stakes, or
    already-decided by the player's own plainly-stated intent). Shares the
    same "outcome" key the narrator already expects (narrate_outcome doesn't
    need a separate code path - it just narrates a certain, uncontested
    action the same as it would narrate any other outcome), plus an "auto"
    flag so the narrator/UI can tell this wasn't a real roll if it matters.
    """
    return {
        "dc": None,
        "roll": None,
        "outcome": "standard_success",
        "auto": True,
    }


#Resolve the stat check by rolling 2d20 and comparing it to the target Difficulty Class (DC)
def resolve_check(stat: int, dc_input, e_modifier: int = 0):
    dc, resolved_tier = resolve_dc_tier(dc_input)

    if resolved_tier == NO_CHECK_TIER:
        # Defensive fallback: resolve_check is not normally called for a
        # no-check action (callers are expected to branch on is_no_check
        # first, see app.py/Main.py), but if it happens anyway, return a
        # clean auto-success instead of rolling against a None DC.
        return auto_success_outcome()

    if isinstance(dc_input, str) and _normalize_tier(dc_input) not in difficulty_classes:
        print(f"Note: unrecognized difficulty tier '{dc_input}', resolved to '{resolved_tier}' (DC {dc}).")

    roll = roll_2d20(stat, e_modifier)
    #Rolls should be rewarded for being above the DC, with a great success being 5 or more above the DC, a standard success being equal to or above the DC, a partial success being within 3 below the DC, and a fail being below the DC.
    if roll >= dc + 5:
        outcome = "great_success"
        print(f"Great Success! Rolled {roll} against DC {dc}.")
    elif roll >= dc:
        outcome = "standard_success"
        print(f"Standard Success! Rolled {roll} against DC {dc}.")
    elif roll >= dc - 3:
        outcome = "partial_success"
        print(f"Partial Success! Rolled {roll} against DC {dc}.")
    elif roll >= dc - 9:
        outcome = "fail"
        print(f"Fail! Rolled {roll} against DC {dc}.")
    else:
        outcome = "Crit fail"
        print(f"Crit Fail! Rolled {roll} against DC {dc}.")

    return {
        "dc": dc,
        "roll": roll,
        "outcome": outcome
    }

difficulty_classes = {
    "low_risk": 10,
    "standard": 15,
    "med_risk": 23,
    "high_risk": 30
}
