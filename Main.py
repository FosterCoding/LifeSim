import os
from typing import Any, Dict

from Dice import resolve_check
from Player import Player
from Narrator import interpret_action, narrate_outcome


def create_player() -> Player:
    print("Create your character")
    name = input("Name: ").strip() or "Player"
    age = int(input("Age: ").strip() or 25)
    month = input("Month: ").strip() or "Jan"
    year = int(input("Year: ").strip() or 2026)
    location = input("Location: ").strip() or "Unknown"

    return Player(
        name=name,
        age=age,
        month=month,
        year=year,
        location=location,
        health=int(input("Health: ").strip() or 70),
        strength=int(input("Strength: ").strip() or 50),
        charisma=int(input("Charisma: ").strip() or 50),
        intelligence=int(input("Intelligence: ").strip() or 50),
        willpower=int(input("Willpower: ").strip() or 50),
        stress=int(input("Stress: ").strip() or 30),
        cash=float(input("Cash: ").strip() or 1000.0),
        occupation=input("Occupation: ").strip() or "",
        background=input("Background: ").strip() or "",
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
