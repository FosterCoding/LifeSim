import random
from typing import Dict, Any
from Player import Player

#Convert 0-100 Stats into a dice roll modifier
def stat_to_modifier(stat):
    if stat < 0 or stat > 100:
        raise ValueError("Stat must be between 0 and 100")
    return (stat - 50) // 10

#Roll 2d20 and add the modifier
def roll_2d20(stat: int, e_modifier: int = 0):
    #Runs the stat_to_modifier function and assigns the return value to the variable 'modifier'.
    modifier = stat_to_modifier(stat)
    roll = random.randint(1, 20) + random.randint(1, 20)
    return roll + modifier + e_modifier

#Resolve the stat check by rolling 2d20 and comparing it to the target Difficulty Class (DC)
def resolve_check(stat: int, dc_input, e_modifier: int = 0):
    # If the AI (or you) passed a string like "med_risk", translate it to the number
    if isinstance(dc_input, str):
        if dc_input not in difficulty_classes:
            raise ValueError(f"Unknown difficulty tier: {dc_input}")
        dc = difficulty_classes[dc_input]
    else:
        dc = dc_input  # If they passed a raw integer, just use it directly

    
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
    else:
        outcome = "fail"
        print(f"Fail! Rolled {roll} against DC {dc}.")

    return {
        "dc": dc,
        "roll": roll,
        "outcome": outcome
    }

difficulty_classes = {
    "standard": 15,
    "med_risk": 23,
    "high_risk": 30
}
