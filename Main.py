import os
from typing import Any, Dict

from Dice import resolve_check
from Player import Player
from Narrator import interpret_action, narrate_outcome, generate_character


def create_player() -> Player:
    print("Create your character")
    mode = input("Random or Custom character? (random/custom): ").strip().lower() or "random"

    try:
        if mode.startswith("c"):
            custom_prompt = input(
                "Describe your character (era, location, age, background, stats, up to 3 relationships): "
            ).strip()
            data = generate_character(custom_prompt) if custom_prompt else generate_character()
        else:
            data = generate_character()
    except Exception as exc:
        print(f"Character generation failed, using defaults: {exc}")
        data = {}

    return Player(
        name=data.get("name", "Player"),
        age=int(data.get("age", 25)),
        month=data.get("month", "Jan"),
        year=int(data.get("year", 2026)),
        location=data.get("location", "Unknown"),
        health=int(data.get("health", 10)),
        strength=int(data.get("strength", 10)),
        charisma=int(data.get("charisma", 10)),
        intelligence=int(data.get("intelligence", 10)),
        willpower=int(data.get("willpower", 10)),
        stress=int(data.get("stress", 0)),
        cash=float(data.get("cash", 1000.0)),
        occupation=data.get("occupation", ""),
        background=data.get("background", ""),
        relationships=data.get("relationships", {}),
    )

def main() -> None:
    print("LifeSim")
    print("Type 'quit' at any prompt to exit.\n")

    player = create_player()

    while True:
        action = input("What do you do? ").strip()
        if action.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        player_state = player.export_engine_state()

        try:
            classification = interpret_action(player_state, action)
        except Exception as exc:
            classification = {
                "stat_used": "charisma",
                "dc_input": "med_risk",
                "e_modifier": 0,
                "reasoning": f"Local fallback: {exc}",
            }

        stat_name = classification.get("stat_used") or "charisma"
        stat_value = getattr(player, stat_name, None)
        if stat_value is None:
            stat_value = player.charisma

        try:
            dice_outcome = resolve_check(
                stat_value,
                classification.get("dc_input", "med_risk"),
                classification.get("e_modifier", 0),
            )
        except Exception as exc:
            dice_outcome = {
                "dc": 23,
                "roll": 0,
                "outcome": f"error: {exc}",
            }

        try:
            narration = narrate_outcome(player_state, action, dice_outcome)
        except Exception as exc:
            narration = {
                "narrative": f"Narration unavailable: {exc}",
                "choices": ["Try again."],
                "state_deltas": {},
            }

        print("\n--- Classification ---")
        print(classification)
        print("\n--- Dice Result ---")
        print(dice_outcome)
        print("\n--- Narration ---")
        print(narration)
        print()


if __name__ == "__main__":
    main()