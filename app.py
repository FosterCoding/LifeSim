import os
import re
import uuid
from flask import Flask, jsonify, render_template, request, session

# Import core game engine modules
from Dice import resolve_check
from Player import Player
from Narrator import interpret_action, narrate_outcome, generate_character, answer_question, generate_bio
import save_store

save_store.init_db()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lifesim_secret_key_12345")

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS

# In-memory session store mapping session_id -> Player object instance
game_sessions = {}
# session_id -> save_id (the stable SQLite row key for this character)
session_save_ids = {}

# Superuser commands: an action wrapped entirely in $...$, e.g. "$Who is Willie Mae$"
# or "$DiceRoll: Crit Success$". These bypass normal action classification and dice
# resolution - Python decides the outcome (if any) up front and hands it to the
# model as a fact to narrate around, rather than asking the model to invent or
# override a dice result itself.
#
# Questions use a SEPARATE syntax, ?...?, e.g. "?Who is Willie Mae?". This is
# intentionally distinct from $...$ rather than relying on the model to guess
# from phrasing whether a $...$ command is a question - a dedicated syntax
# removes that ambiguity in Python before the model ever sees the turn.
SUPERUSER_PATTERN = re.compile(r"^\$(.+)\$$", re.DOTALL)
QUESTION_PATTERN = re.compile(r"^\?(.+)\?$", re.DOTALL)
DICE_OVERRIDE_PATTERN = re.compile(r"^dice\s*roll\s*:\s*(.+)$", re.IGNORECASE)

# Maps free-text override requests to the exact outcome strings Dice.py produces,
# so an override is indistinguishable downstream from a real roll landing on that tier.
DICE_OVERRIDE_OUTCOMES = {
    "great success": "great_success",
    "crit success": "great_success",
    "critical success": "great_success",
    "success": "standard_success",
    "standard success": "standard_success",
    "partial success": "partial_success",
    "partial": "partial_success",
    "fail": "fail",
    "failure": "fail",
    "crit fail": "Crit fail",
    "critical fail": "Crit fail",
    "critical failure": "Crit fail",
}


def parse_superuser_command(action_text: str):
    """
    Returns None if action_text is a normal in-fiction action.
    Otherwise returns one of:
      {"type": "question", "raw": <inner text>}          - ?...? syntax, answer only, no narrative
      {"type": "dice_override", "outcome": <tier>, "raw": <inner text>}
      {"type": "command", "raw": <inner text>}            - any other $...$ (state changes, misc out-of-character asks)
    """
    stripped = action_text.strip()

    question_match = QUESTION_PATTERN.match(stripped)
    if question_match:
        return {"type": "question", "raw": question_match.group(1).strip()}

    match = SUPERUSER_PATTERN.match(stripped)
    if not match:
        return None

    inner = match.group(1).strip()
    dice_match = DICE_OVERRIDE_PATTERN.match(inner)
    if dice_match:
        requested = dice_match.group(1).strip().lower()
        outcome = DICE_OVERRIDE_OUTCOMES.get(requested, "standard_success")
        return {"type": "dice_override", "outcome": outcome, "raw": inner}

    return {"type": "command", "raw": inner}


# Time skip commands: &y/&m/&w/&d/&h optionally prefixed with a number, e.g.
# "&m", "&2m", "&3y", found anywhere in the action text (usually at the start
# or end). Python parses this and computes the exact resulting date via
# Player.advance_date() before the model is ever called - the model never
# invents or increments a date itself, closing the drift bug where &2m then
# &m produced two independently-guessed dates instead of one continuous timeline.
TIME_SKIP_PATTERN = re.compile(r"&(\d*)([ymwdh])\b", re.IGNORECASE)


def parse_time_skip(action_text: str):
    """
    Returns None if no time-skip token is present. Otherwise returns
    (unit, amount, cleaned_action_text) where cleaned_action_text has the
    &-token removed so it doesn't leak into the narrative prompt as literal
    text the model has to interpret itself.
    """
    match = TIME_SKIP_PATTERN.search(action_text)
    if not match:
        return None

    amount_str, unit = match.groups()
    amount = int(amount_str) if amount_str else 1
    unit = unit.lower()

    cleaned = (action_text[:match.start()] + action_text[match.end():]).strip()
    return unit, amount, cleaned


def get_current_player():
    """Helper to retrieve or initialize the Player object for the current session."""
    session_id = session.get("session_id")
    if session_id and session_id in game_sessions:
        return game_sessions[session_id]
    return None


def get_current_save_id():
    session_id = session.get("session_id")
    return session_save_ids.get(session_id)


@app.route("/")
def index():
    """Serves the main web interface."""
    return render_template("index.html")


def _build_start_date(month_str: str, year: int, day: int = 1):
    """
    Converts a month name (e.g. 'January', 'Jan') + year into a real date for
    a new character's starting point. Defaults to the 1st of the month, and
    falls back to January if the month string doesn't parse rather than
    raising, since a wrong starting day is a much smaller problem than the
    whole character-creation flow crashing.
    """
    from datetime import date as _date
    month_names_full = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    month_names_abbr = [m[:3] for m in month_names_full]
    normalized = str(month_str).strip().lower()

    if normalized in month_names_full:
        month_num = month_names_full.index(normalized) + 1
    elif normalized in month_names_abbr:
        month_num = month_names_abbr.index(normalized) + 1
    else:
        month_num = 1

    try:
        return _date(int(year), month_num, day)
    except ValueError:
        return _date(int(year), month_num, 1)


def _clamped_stat(raw_value, default: int = 10) -> int:
    """
    Parses a stat value (from a request body or AI-generated character data)
    and clamps it to the valid 0-20 range, the same range Player.apply_deltas
    already enforces for in-game stat changes. Creation was the one path that
    never enforced this, letting a manually-typed or AI-hallucinated value
    like 999 slip straight into Player unclamped. Falls back to default on
    anything unparseable rather than raising, consistent with how the rest of
    character creation degrades gracefully instead of failing the whole request.
    """
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0, min(20, value))


def _format_hour(hour: int) -> str:
    """12-hour clock format with AM/PM for a raw hour int (0-23), matching
    Player.formatted_time's logic - used here to format a captured 'before'
    hour value without needing a full Player instance."""
    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:00 {period}"


@app.route("/api/generate-draft", methods=["POST"])
def generate_draft():
    """
    Generates a random character via the AI, same as the 'random' new-game mode,
    but does NOT create a Player, session, or save. Returns the raw generated
    fields so the creation form can be pre-filled and edited before the player
    actually commits with Begin.
    """
    data = request.get_json(silent=True) or {}
    try:
        gen = generate_character(data.get("prompt"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"draft": gen})


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Creates a new character instance and generates the starting scene."""
    data = request.get_json(silent=True) or {}

    mode = data.get("mode", "manual")
    if mode in ("random", "custom"):
        try:
            gen = generate_character(data.get("prompt") if mode == "custom" else None)
        except Exception:
            gen = {}
        start_date = _build_start_date(gen.get("month", "January"), gen.get("year", 2026))
        preset = data.get("difficulty_preset", "standard")
        if preset not in ("gritty", "standard", "forgiving"):
            preset = "standard"
        player = Player(
            name=gen.get("name", "Player"),
            age=int(gen.get("age", 25)),
            date=start_date,
            location=gen.get("location", "New York, USA"),
            race=gen.get("race", ""),
            health=_clamped_stat(gen.get("health", 10)),
            strength=_clamped_stat(gen.get("strength", 10)),
            charisma=_clamped_stat(gen.get("charisma", 10)),
            intelligence=_clamped_stat(gen.get("intelligence", 10)),
            willpower=_clamped_stat(gen.get("willpower", 10)),
            stress=_clamped_stat(gen.get("stress", 0), default=0),
            cash=float(gen.get("cash", 1000.0)),
            occupation=gen.get("occupation", "Unemployed"),
            background=gen.get("background", "Looking for a fresh start."),
            relationships=gen.get("relationships", {}),
            last_paycheck_date=start_date,
            difficulty_preset=preset,
            job_title=gen.get("job_title", ""),
            salary=float(gen.get("salary", 0.0)),
            pay_frequency=gen.get("pay_frequency", "biweekly"),
        )
        generated_inventory = gen.get("inventory")
        if isinstance(generated_inventory, list):
            player.inventory = [str(item).strip() for item in generated_inventory if str(item).strip()][:10]
        generated_expenses = gen.get("expenses")
        if isinstance(generated_expenses, list):
            for exp in generated_expenses[:2]:
                if isinstance(exp, dict) and exp.get("name"):
                    player.add_expense(
                        name=str(exp["name"]),
                        amount=float(exp.get("amount", 0.0)),
                        frequency=exp.get("frequency", "monthly"),
                    )
    else:
        # Standard default values or custom inputs sent from UI creation form
        start_date = _build_start_date(data.get("month", "January"), data.get("year", 2026))
        preset = data.get("difficulty_preset", "standard")
        if preset not in ("gritty", "standard", "forgiving"):
            preset = "standard"
        player = Player(
            name=data.get("name", "Player"),
            age=int(data.get("age", 25)),
            date=start_date,
            location=data.get("location", "New York, USA"),
            race=data.get("race", ""),
            health=_clamped_stat(data.get("health", 10)),
            strength=_clamped_stat(data.get("strength", 10)),
            charisma=_clamped_stat(data.get("charisma", 10)),
            intelligence=_clamped_stat(data.get("intelligence", 10)),
            willpower=_clamped_stat(data.get("willpower", 10)),
            stress=_clamped_stat(data.get("stress", 0), default=0),
            cash=float(data.get("cash", 1000.0)),
            occupation=data.get("occupation", "Unemployed"),
            background=data.get("background", "Looking for a fresh start."),
            last_paycheck_date=start_date,
            difficulty_preset=preset,
        )
        inventory = data.get("inventory")
        if isinstance(inventory, list):
            player.inventory = [str(item).strip() for item in inventory if str(item).strip()][:10]
        relationships = data.get("relationships")
        if isinstance(relationships, dict):
            player.relationships = dict(list(relationships.items())[:3])
        expense_name = data.get("expense_name")
        if expense_name:
            try:
                expense_amount = float(data.get("expense_amount", 0.0))
            except (TypeError, ValueError):
                expense_amount = 0.0
            if expense_amount > 0:
                player.add_expense(
                    name=str(expense_name).strip(),
                    amount=expense_amount,
                    frequency=data.get("expense_frequency", "monthly"),
                )

    # Starting relationships (from either the AI generator or manual entry)
    # go through the same 0-20 normalization as everything else, since
    # neither creation path otherwise clamps quality before it lands on Player.
    player.normalize_relationship_scales()

    # Assign a session token to the web user
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    game_sessions[session_id] = player

    save_id = str(uuid.uuid4())
    session_save_ids[session_id] = save_id
    save_store.save_playthrough(save_id, player.to_save_dict())

    player_state = player.export_engine_state()

    # Opening scene prompt (shown once the player taps Next past the summary screen)
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

    # Bio for the character-creation summary screen. Deliberately a separate
    # call from the opening scene above - previously the summary screen just
    # displayed that same opening narration, so the player read identical
    # text twice (once on the summary screen, again as the first real scene).
    try:
        bio = generate_bio(player_state)
    except Exception:
        bio = player_state.get("background") or "No background available."

    return jsonify(
        {
            "player_state": player_state,
            "narration": narration,
            "bio": bio,
            "save_id": save_id,
        }
    )


@app.route("/api/action", methods=["POST"])
def perform_action():
    """Processes a turn: interprets user input, resolves dice math, and narrates results."""
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active game session found. Please start a new game."}), 400

    data = request.get_json(silent=True) or {}
    action_text = data.get("action", "").strip()

    if not action_text:
        return jsonify({"error": "Action cannot be empty."}), 400

    # Optional client-side cheat lock: while active, every normal action this
    # turn skips the real dice roll and forces this outcome instead. This is
    # separate from the one-shot $DiceRoll: ...$ superuser syntax - a locked
    # outcome still goes through real classification/narration, only the roll
    # itself is bypassed, so the scene and choices continue normally.
    locked_outcome_label = data.get("locked_outcome")
    locked_outcome = None
    if locked_outcome_label:
        normalized = str(locked_outcome_label).strip().lower()
        locked_outcome = DICE_OVERRIDE_OUTCOMES.get(normalized)

    # Detect and apply any &-prefixed time skip BEFORE capturing player_state,
    # so the model sees the already-advanced date on this same turn rather
    # than narrating against a stale one. The &-token itself is stripped out
    # so it doesn't leak into the action text sent to the model.
    time_skip_info = None
    skip_match = parse_time_skip(action_text)
    if skip_match:
        unit, amount, action_text = skip_match
        try:
            old_hour = player.hour
            old_date, new_date = player.advance_date(unit, amount)
            time_skip_info = {
                "unit": unit,
                "amount": amount,
                "from": old_date.strftime("%m/%d/%Y"),
                "to": new_date.strftime("%m/%d/%Y"),
                "from_time": _format_hour(old_hour),
                "to_time": _format_hour(player.hour),
            }
        except ValueError as exc:
            return jsonify({"error": f"Invalid time skip: {exc}"}), 400

        # Process any paydays that fall within the skipped span. Cash already
        # lands on the Player object here; time_skip_info just carries the
        # summary through so the narrator can mention it naturally.
        paychecks = player.process_payday()
        if paychecks:
            time_skip_info["paychecks"] = paychecks

        # Symmetrical to paydays: deduct any recurring expenses (rent, bills,
        # debt payments) that came due within the skipped span.
        expense_payments = player.process_expenses()
        if expense_payments:
            time_skip_info["expense_payments"] = expense_payments

        # NPC background agency: every ~30 in-fiction days, a probabilistic
        # subset of known NPCs are flagged as due for a background life
        # update, independent of whether the player actually visited them.
        drifted_npcs = player.check_npc_drift()
        if drifted_npcs:
            time_skip_info["drifted_npcs"] = drifted_npcs

        if not action_text:
            # A bare time skip with no other action text (e.g. just "&2m")
            # still needs something for the narrator to react to.
            action_text = "Time passes."

    player_state = player.export_engine_state()

    superuser = parse_superuser_command(action_text)

    if superuser:
        # Superuser commands never touch action classification or the dice engine.
        # Python decides everything up front; the model only narrates around it.
        classification = {
            "stat_used": None,
            "dc_input": None,
            "e_modifier": 0,
            "reasoning": "Superuser command - classification bypassed.",
        }

        if superuser["type"] == "question":
            # A pure ?...? question never touches narration, dice, or state.
            # It gets a direct answer and nothing else - no new scene, no
            # invented choices, no state_deltas.
            try:
                answer = answer_question(player_state, superuser["raw"])
            except Exception as exc:
                answer = f"[Could not answer: {exc}]"

            narration = {
                "narrative": answer,
                "choices": [],
                "state_deltas": {},
            }
            dice_outcome = {"outcome": "superuser_question", "superuser_override": True}

        else:
            if superuser["type"] == "dice_override":
                dice_outcome = {"outcome": superuser["outcome"], "superuser_override": True}
            else:
                dice_outcome = {"outcome": "superuser_command", "superuser_override": True}

            try:
                narration = narrate_outcome(
                    player_state,
                    action_text,
                    dice_outcome,
                    superuser_command=superuser["raw"],
                    time_skip=time_skip_info,
                )
            except Exception as exc:
                narration = {
                    "narrative": f"[Superuser command received, but narration failed: {exc}]",
                    "choices": ["Continue"],
                    "state_deltas": {},
                }

    else:
        # 1. Interpret user input via LLM (also estimates elapsed time)
        try:
            classification = interpret_action(player_state, action_text)
        except Exception as exc:
            classification = {
                "stat_used": "charisma",
                "dc_input": "med_risk",
                "e_modifier": 0,
                "reasoning": f"Local fallback: {exc}",
                "elapsed_unit": "h",
                "elapsed_amount": 1,
            }

        # 1b. Advance the calendar from the AI's elapsed-time estimate, but
        # only if the player didn't already supply an explicit &-command this
        # turn (time_skip_info would already be set in that case). This closes
        # the gap where choosing a normal action/choice implying real duration
        # ("lay low for a few weeks") never moved the tracked date at all,
        # since previously only explicit &-tags triggered advance_date().
        if time_skip_info is None:
            est_unit = classification.get("elapsed_unit", "h")
            est_amount = classification.get("elapsed_amount", 1)
            try:
                old_hour = player.hour
                old_date, new_date = player.advance_date(est_unit, est_amount)
                if old_date != new_date:
                    time_skip_info = {
                        "unit": est_unit,
                        "amount": est_amount,
                        "from": old_date.strftime("%m/%d/%Y"),
                        "to": new_date.strftime("%m/%d/%Y"),
                        "from_time": _format_hour(old_hour),
                        "to_time": _format_hour(player.hour),
                        "estimated": True,  # distinguishes an AI estimate from an explicit player &-command
                    }
                    paychecks = player.process_payday()
                    if paychecks:
                        time_skip_info["paychecks"] = paychecks
                    expense_payments = player.process_expenses()
                    if expense_payments:
                        time_skip_info["expense_payments"] = expense_payments
                    drifted_npcs = player.check_npc_drift()
                    if drifted_npcs:
                        time_skip_info["drifted_npcs"] = drifted_npcs
                    # Refresh player_state so narrate_outcome (below) sees the
                    # already-advanced date this same turn, not a stale one.
                    player_state = player.export_engine_state()
            except ValueError:
                pass  # malformed unit from the model; leave the date untouched rather than fail the turn

        # 2. Extract stat value from the Player instance
        stat_name = classification.get("stat_used") or "charisma"
        stat_value = getattr(player, stat_name, None)
        if stat_value is None:
            stat_value = player.charisma

        # 3. Resolve the 2d20 check using Dice engine, unless a cheat lock is
        # forcing the outcome for this turn instead.
        if locked_outcome:
            dice_outcome = {"outcome": locked_outcome, "superuser_override": True, "locked": True}
        else:
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
            narration = narrate_outcome(player_state, action_text, dice_outcome, time_skip=time_skip_info)
        except Exception as exc:
            narration = {
                "narrative": f"Action complete. (Narration error: {exc})",
                "choices": ["Continue", "Look around"],
                "state_deltas": {},
            }

    # 5. Apply state changes using Player's built-in delta engine
    deltas = narration.get("state_deltas", {})
    if isinstance(deltas, dict):
        player.apply_deltas(deltas)

    # 6. Auto-save this turn
    save_id = get_current_save_id()
    if save_id:
        try:
            save_store.save_playthrough(save_id, player.to_save_dict())
        except Exception as exc:
            print(f"Auto-save failed (game continues normally): {exc}")

    return jsonify(
        {
            "classification": classification,
            "dice_outcome": dice_outcome,
            "narration": narration,
            "player_state": player.export_engine_state(),
        }
    )


@app.route("/api/saves", methods=["GET"])
def list_saves():
    """Lists every saved playthrough for the 'Load Game' browser in the UI."""
    try:
        saves = save_store.list_playthroughs()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"saves": saves})


@app.route("/api/resume", methods=["POST"])
def resume_game():
    """
    Resumes an existing playthrough. If save_id is provided, loads that specific
    character (used by the saves browser). If not, and this browser's session
    already has an active game, just returns the current state as-is (used by
    the browser-refresh fix and the 'Continue Life' button, so a refresh never
    loses progress that's already sitting in game_sessions/SQLite).
    """
    data = request.get_json(silent=True) or {}
    requested_save_id = data.get("save_id")

    if requested_save_id:
        record = save_store.load_playthrough(requested_save_id)
        if not record:
            return jsonify({"error": "Save not found."}), 404
        try:
            player = Player.from_save_dict(record)
        except Exception as exc:
            # A corrupted or malformed save record (bad date string, unexpected
            # type, etc.) should never crash the whole request with a raw 500 -
            # that returns an HTML error page the frontend can't parse into a
            # useful message, and previously looked identical to a hang.
            return jsonify({"error": f"This save appears corrupted and can't be loaded: {exc}"}), 500

        # One-time repair for saves created before relationship quality and
        # reputation were clamped to 0-20 - pulls any out-of-range values
        # (e.g. quality: 35) back into range immediately on load rather than
        # leaving them broken until an unrelated delta happens to touch that
        # specific NPC or group again.
        if player.normalize_relationship_scales():
            try:
                save_store.save_playthrough(requested_save_id, player.to_save_dict())
            except Exception:
                pass  # repair still applies to this session even if the persist fails

        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        game_sessions[session_id] = player
        session_save_ids[session_id] = requested_save_id

        return jsonify({"player_state": player.export_engine_state(), "save_id": requested_save_id})

    # No save_id given: try to resume whatever this browser session already has.
    player = get_current_player()
    if player:
        return jsonify({
            "player_state": player.export_engine_state(),
            "save_id": get_current_save_id(),
        })

    return jsonify({"error": "No active session to resume."}), 404


@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns the current engine state for the active player."""
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active session"}), 404
    return jsonify(player.export_engine_state())


@app.route("/api/set-speed", methods=["POST"])
def set_speed():
    """
    Directly sets the player's persistent sim_speed. This is a pure settings
    change: no AI call, no narration, no auto-saved turn - flipping the pace
    should not itself cost anything or advance the story. Also auto-saves the
    updated preference so it survives a reload/resume like everything else.
    """
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active game session found."}), 400

    data = request.get_json(silent=True) or {}
    speed = data.get("speed")
    if speed not in ("h", "d", "w", "m", "y"):
        return jsonify({"error": "Invalid speed value."}), 400

    player.sim_speed = speed

    save_id = get_current_save_id()
    if save_id:
        try:
            save_store.save_playthrough(save_id, player.to_save_dict())
        except Exception as exc:
            print(f"Auto-save after speed change failed (setting still applies this session): {exc}")

    return jsonify({"player_state": player.export_engine_state()})


if __name__ == "__main__":
    # Local dev server running on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
