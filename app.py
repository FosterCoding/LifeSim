#Flask
import os
import uuid
from flask import Flask, jsonify, render_template, request, session

# Import core game engine modules without touching them
from Dice import resolve_check
from Player import Player
from Narrator import interpret_action, narrate_outcome

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lifesim_secret_key_12345")

# In-memory session store mapping session_id -> Player object instance
game_sessions = {}


def get_current_player():
    """Helper to retrieve or initialize the Player object for the current session."""
    session_id = session.get("session_id")
    if session_id and session_id in game_sessions:
        return game_sessions[session_id]
    return None


@app.route("/")
def index():
    """Serves the main web interface."""
    return render_template("index.html")


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Creates a new character instance and generates the starting scene."""
    data = request.get_json(silent=True) or {}

    # Standard default values or custom inputs sent from UI creation form
    player = Player(
        name=data.get("name", "Player"),
        age=int(data.get("age", 25)),
        month=data.get("month", "Jan"),
        year=int(data.get("year", 2026)),
        location=data.get("location", "New York, USA"),
        health=int(data.get("health", 70)),
        strength=int(data.get("strength", 50)),
        charisma=int(data.get("charisma", 50)),
        intelligence=int(data.get("intelligence", 50)),
        willpower=int(data.get("willpower", 50)),
        stress=int(data.get("stress", 30)),
        cash=float(data.get("cash", 1000.0)),
        occupation=data.get("occupation", "Unemployed"),
        background=data.get("background", "Looking for a fresh start."),
    )

    # Assign a session token to the web user
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    game_sessions[session_id] = player

    player_state = player.export_engine_state()

    # Opening scene prompt
    initial_action = "Character creation complete. Introduce the character's starting situation."
    try:
        narration = narrate_outcome(
            player_state,
            initial_action,
            {"dc": 0, "roll": 0, "outcome": "start"},
        )
    except Exception as exc:
        narration = {
            "narrative": f"Welcome to the life simulation! (Fallback scene: {exc})",
            "choices": ["Look around.", "Check your pocket.", "Find a job."],
            "state_deltas": {},
        }

    return jsonify(
        {
            "player_state": player_state,
            "narration": narration,
        }
    )


@app.route("/api/action", methods=["POST"])
def perform_action():
    """Processes a turn: interprets user input, resolves dice math, and narrate results."""
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active game session found. Please start a new game."}), 400

    data = request.get_json(silent=True) or {}
    action_text = data.get("action", "").strip()

    if not action_text:
        return jsonify({"error": "Action cannot be empty."}), 400

    player_state = player.export_engine_state()

    # 1. Interpret user input via LLM
    try:
        classification = interpret_action(player_state, action_text)
    except Exception as exc:
        classification = {
            "stat_used": "charisma",
            "dc_input": "med_risk",
            "e_modifier": 0,
            "reasoning": f"Local fallback: {exc}",
        }

    # 2. Extract stat value from the Player instance
    stat_name = classification.get("stat_used") or "charisma"
    stat_value = getattr(player, stat_name, None)
    if stat_value is None:
        stat_value = player.charisma

    # 3. Resolve the 2d20 check using Dice engine
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

    # 4. Generate story outcome via LLM
    try:
        narration = narrate_outcome(player_state, action_text, dice_outcome)
    except Exception as exc:
        narration = {
            "narrative": f"Action complete. (Narration error: {exc})",
            "choices": ["Continue", "Look around"],
            "state_deltas": {},
        }

    # 5. Apply state changes returned in narration state_deltas to Player instance
    deltas = narration.get("state_deltas", {})
    if isinstance(deltas, dict):
        for key, value in deltas.items():
            if hasattr(player, key):
                current_val = getattr(player, key)
                if isinstance(current_val, (int, float)) and isinstance(value, (int, float)):
                    setattr(player, key, current_val + value)
                elif isinstance(current_val, list) and isinstance(value, list):
                    setattr(player, key, current_val + value)
                elif isinstance(current_val, dict) and isinstance(value, dict):
                    current_val.update(value)

    return jsonify(
        {
            "classification": classification,
            "dice_outcome": dice_outcome,
            "narration": narration,
            "player_state": player.export_engine_state(),
        }
    )


@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns the current engine state for the active player."""
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active session"}), 404
    return jsonify(player.export_engine_state())


if __name__ == "__main__":
    # Local dev server running on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)