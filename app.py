import os
import re
import time
import uuid
import json
import threading
from flask import Flask, jsonify, render_template, request, session, Response

# Import core game engine modules
from Dice import resolve_check, is_no_check, auto_success_outcome
from Player import Player
from Narrator import interpret_action, narrate_outcome, generate_character, answer_question, generate_bio, extract_mechanical_deltas, plan_scene, write_scene, truncate_scene_if_needed, stream_bio, expand_background
import save_store

save_store.init_db()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lifesim_secret_key_12345")

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS

# How many consecutive turns, after a player-initiated pace change, the
# model is given to naturally report same_scene_continuation=false on its
# own before Python stops trusting that judgment and forces the pace
# change regardless (see _apply_sim_speed_floor's force_ignore_same_scene
# and _finalize_turn's streak tracking). 1 means: the turn right after a
# pace change gets one real chance to wrap up naturally; if it's STILL
# same-scene next turn, that next turn's floor is forced. Kept low
# deliberately - prompt-only versions of this already failed with an
# unbounded number of attempts (the model never stopped on its own), so a
# generous grace period would just reproduce the same failure more slowly.
_SAME_SCENE_GRACE_TURNS = 1

# In-memory session store mapping session_id -> Player object instance
game_sessions = {}
# session_id -> save_id (the stable SQLite row key for this character)
session_save_ids = {}
# session_id -> threading.Lock, held briefly whenever a background thread is
# mutating that session's live Player object (currently: async mechanical
# extraction after a turn). A normal request acquires this before touching
# player state and releases it when done, so a same-session request that
# somehow arrives while a background mutation is still in flight waits its
# turn instead of reading/writing player state mid-mutation. Created lazily
# per session_id the first time it's needed.
session_locks = {}


def _get_session_lock(session_id: str) -> threading.Lock:
    if session_id not in session_locks:
        session_locks[session_id] = threading.Lock()
    return session_locks[session_id]

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
    """
    Helper to retrieve or initialize the Player object for the current
    session.

    On a multi-worker deployment (PythonAnywhere's paid tiers run more than
    one worker process by default), each worker has its OWN copy of
    game_sessions - it's a plain in-memory dict, not shared across
    processes. A request that creates a session on worker A and a
    following request that lands on worker B would previously find nothing
    in worker B's game_sessions at all, even though the player's save_id
    cookie correctly round-tripped through the browser - this was the exact
    cause of "No active game session found" firing on the very first action
    after character creation, and "Could not generate a fresh scene" on
    resume.

    The fix: save_id itself is stored directly in the Flask session cookie
    (see new_game/resume_game below), which the browser sends on every
    request regardless of which worker handles it - no shared server memory
    required. game_sessions is kept only as a same-worker cache to avoid
    re-hitting SQLite on every single call within one worker's lifetime;
    it is never the only place a session can be found. If the session_id
    isn't in this worker's cache, this always falls back to loading fresh
    from SQLite via save_id and re-populating the cache, rather than
    returning None and forcing the player to start over.
    """
    session_id = session.get("session_id")
    save_id = session.get("save_id")

    if session_id and session_id in game_sessions:
        cached_player = game_sessions[session_id]

        # A cache hit alone is not enough to trust this worker's copy. On a
        # multi-worker deployment, game_sessions is a per-worker dict that
        # is only ever written, never invalidated - if THIS session has
        # been handled by a DIFFERENT worker any time since this worker
        # last cached it, this worker's copy can be arbitrarily behind
        # (missing turns' worth of date advancement, cash changes, choice-
        # cadence state, everything). Previously this went completely
        # unchecked: a stale cached Player would silently keep being used,
        # its own advance_date calls would build FORWARD from an already-
        # stale baseline, and the resulting wrong state would be returned
        # to the player even though save_playthrough's turn_number guard
        # correctly stopped it from ever overwriting the newer SQLite row -
        # the save was protected, but the RESPONSE sent back never was.
        # This surfaced as things like the displayed date visibly rolling
        # backward turn to turn depending on which worker happened to
        # handle each request.
        #
        # The check costs one SQLite read on every cache hit (there is no
        # cheaper column-only lookup available - save_store stores the
        # whole player as one JSON blob), which is an accepted tradeoff for
        # correctness: a stale player object is worse than one extra read
        # per turn, and this only ever adds a read, it never removes the
        # in-memory fast path's benefit of avoiding a full Player
        # reconstruction on the common (not-stale) case below.
        if save_id:
            try:
                fresh_data = save_store.load_playthrough(save_id)
            except Exception:
                fresh_data = None  # fall through to trusting the cache rather than failing the request over this

            if fresh_data is not None:
                fresh_turn = fresh_data.get("turn_number", 0)
                cached_turn = getattr(cached_player, "turn_number", 0)
                if fresh_turn > cached_turn:
                    # SQLite has moved ahead of this worker's cache - some
                    # other worker handled a more recent turn. Rebuild from
                    # the fresh data instead of trusting the stale object.
                    cached_player = Player.from_save_dict(fresh_data)
                    game_sessions[session_id] = cached_player

        return cached_player

    if not save_id:
        return None

    try:
        data = save_store.load_playthrough(save_id)
    except Exception as exc:
        print(f"Note: session cache miss and SQLite fallback load failed: {exc}")
        return None

    if not data:
        return None

    player = Player.from_save_dict(data)

    # Re-populate this worker's cache and the legacy session_save_ids map
    # (still used by get_current_save_id/_apply_extraction_async) so the
    # rest of this worker's requests for the same session don't need to
    # hit SQLite again.
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
    game_sessions[session_id] = player
    session_save_ids[session_id] = save_id

    return player


def get_current_save_id():
    """
    save_id lives directly in the Flask session cookie now (see
    get_current_player's docstring for why) - session_save_ids is kept only
    as a same-worker cache and is never the sole source of truth, so this
    always checks the cookie first.
    """
    save_id = session.get("save_id")
    if save_id:
        return save_id
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


def _apply_sim_speed_floor(est_unit: str, est_amount: int, sim_speed: str, same_scene_continuation: bool = False, force_ignore_same_scene: bool = False):
    """
    sim_speed is supposed to be a persistent pacing preference the player
    sets once and expects to stick ("Weekly" should mean routine turns
    advance about a week at a time until changed) - but it was previously
    pure prompt guidance with nothing in Python enforcing it, so the
    classifier's own per-turn elapsed_unit/elapsed_amount estimate could
    just... not use it, turn after turn, with no way to tell from outside
    whether that was a deliberate quick-beat call or the model simply
    defaulting to "h" out of habit. This makes sim_speed an actual floor:
    if the classifier's estimate comes in shorter than the player's set
    pace, bump it up to match.

    No exception for "auto"-tier turns as a CATEGORY (deliberate choice -
    see project history): an earlier version exempted auto turns so a
    routine, uncontested action could still stay at hour-scale even under a
    slower sim_speed. In practice this meant nearly every turn skipped the
    floor entirely, since most ordinary choices correctly classify as auto
    - so sim_speed: Day/Week was silently never respected.

    same_scene_continuation IS respected here (unless force_ignore_same_scene
    overrides it, see below), and is a narrower, different case than the
    rejected "auto" exception above - it is not "was this turn low-stakes,"
    it's "is this turn still inside the exact same continuous beat the
    previous scene was already narrating" (the next line of a conversation,
    the next moment of a journey already underway). Without this, a
    multi-turn continuous scene (e.g. "find a seat on the train" -> "look
    out the window" -> "arrive") got sim_speed's full floor applied to
    EVERY beat of it under a Day/Week/Month pace, silently fast-forwarding
    the calendar by days between beats of a scene the narrator was
    simultaneously depicting as one uninterrupted afternoon - a real,
    confirmed bug (the tracked date and the narrated scene visibly
    disagreed with each other, turn over turn).

    force_ignore_same_scene exists for a different, later-discovered
    problem: when the PLAYER changes sim_speed mid-scene, the model was
    given explicit instructions (both a hard-cut version and a softer
    two-stage "conclude, then switch" version) to wrap up and move on -
    and real production logs confirmed both were correctly delivered and
    simply not complied with; the model kept reporting
    same_scene_continuation=true and narrating the same scene indefinitely.
    app.py's _finalize_turn tracks how many consecutive turns this
    persists after a pace change and, once a grace period is exhausted,
    calls this with force_ignore_same_scene=True - at that point the
    model's own same_scene_continuation judgment is no longer trusted for
    this decision, and the floor applies regardless of what it reports.
    This is a deliberate last resort: it can cut a scene less gracefully
    than the player might want, but it guarantees the pace setting is
    never silently ignored forever, which prompt instructions alone
    failed to guarantee.

    Explicit player &-commands are NOT handled here - they're protected one
    level up, at the call site, which only invokes this function when
    time_skip_info is still None (i.e. no explicit &-command already set
    it this turn). An &-command therefore never reaches this function at
    all, regardless of dc_input.
    """
    if same_scene_continuation and not force_ignore_same_scene:
        return est_unit, est_amount

    UNIT_TO_HOURS = {"h": 1, "d": 24, "w": 24 * 7, "m": 24 * 30, "y": 24 * 365}

    sim_speed_hours = UNIT_TO_HOURS.get(sim_speed, 24)
    est_hours = UNIT_TO_HOURS.get(est_unit, 1) * max(est_amount, 1)

    if est_hours < sim_speed_hours:
        return sim_speed, 1
    return est_unit, est_amount


def _apply_extraction_async(player, session_id: str, save_id: str, player_state: dict, narrative_text: str) -> None:
    """
    Runs the mechanical extraction pass (Narrator.extract_mechanical_deltas)
    and applies its result to the live player, entirely in a background
    thread AFTER the player's response has already been sent. This is safe
    because nothing in the response the player is looking at depends on
    extraction's outcome - it only ever affects data on the Crew,
    Businesses, and Relationships tabs, which the player checks separately,
    never the scene text or choices they're reading right now.

    Acquires this session's lock for the whole apply-and-save sequence, so
    a same-session request that happens to arrive while this is still
    running (a fast double action) waits its turn rather than reading or
    writing player state mid-mutation. Performs its own save at the end,
    since the main request's auto-save may already have fired before this
    finishes - without a follow-up save here, whatever extraction adds
    would only live in memory until the next turn's save, and could be
    lost entirely if the app restarted in between. save_store's
    turn_number-ordering guard makes this save safe even if it lands after
    a genuinely newer save from a later turn - the stale write is simply
    rejected rather than silently overwriting newer state (see
    save_store.save_playthrough).

    KNOWN, ACCEPTED GAP: the session lock above only coordinates threads
    within THIS worker process. On a multi-worker deployment, if the
    player's very next action lands on a DIFFERENT worker before this
    function finishes and saves, that worker will load this session fresh
    from SQLite (see get_current_player's fallback) and may not yet see
    this turn's extracted crew/business/rival changes. The window is narrow
    (roughly one small LLM call's duration) and self-heals on the next save
    - the missing data isn't lost, just briefly not-yet-written. Note this
    gap is now strictly about extraction data being briefly absent, not
    about stale data ever winning a save race - the turn_number guard
    closes that half of the problem entirely.
    """
    lock = _get_session_lock(session_id) if session_id else threading.Lock()
    with lock:
        try:
            extracted = extract_mechanical_deltas(player_state, narrative_text)
            player.apply_extracted_deltas(extracted)
        except Exception as exc:
            print(f"Note: background mechanical extraction failed (game continues normally): {exc}")
            return

        if save_id:
            try:
                save_store.save_playthrough(save_id, player.to_save_dict())
            except Exception as exc:
                print(f"Note: background save after extraction failed (game continues normally): {exc}")



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


def _build_new_character(data: dict) -> Player:
    """
    Builds and returns a fully-configured Player from new-game request data
    - both the AI-generated (random/custom) and manual creation paths.
    Extracted from new_game() so the new streaming route
    (/api/new-game/stream) can share the exact same character-construction
    logic rather than duplicating it; only session/save assignment and the
    opening scene generation differ between the two routes.
    """
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
        genre = data.get("genre", "realism")
        if genre not in ("realism", "fantasy", "horror"):
            genre = "realism"
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
            genre=genre,
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
        genre = data.get("genre", "realism")
        if genre not in ("realism", "fantasy", "horror"):
            genre = "realism"

        raw_background = data.get("background", "").strip()
        # Only expanded when the player explicitly asked for it (the
        # custom-form "Expand with AI" toggle) - previously a typed
        # background was always used completely verbatim with no way to
        # develop it into a fuller backstory, which is what a player
        # typing "grew up poor, father was a miner" into a form field
        # would reasonably expect to happen, not have that exact sentence
        # become their entire background. Opt-in, not automatic: someone
        # who wrote their own full backstory and wants it used exactly as
        # written should not have it silently rewritten out from under them.
        if raw_background and data.get("expand_background"):
            try:
                background = expand_background(raw_background, {
                    "name": data.get("name"),
                    "age": data.get("age"),
                    "location": data.get("location"),
                    "occupation": data.get("occupation"),
                    "year": data.get("year"),
                })
            except Exception:
                background = raw_background
        else:
            background = raw_background or "Looking for a fresh start."

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
            background=background,
            last_paycheck_date=start_date,
            difficulty_preset=preset,
            genre=genre,
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

    return player


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Creates a new character instance and generates the starting scene."""
    data = request.get_json(silent=True) or {}

    player = _build_new_character(data)

    # Assign a session token to the web user
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    game_sessions[session_id] = player

    save_id = str(uuid.uuid4())
    session["save_id"] = save_id
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

    # Seed the rolling narrative buffer with the opening scene itself, so
    # the player's very first real action has real continuity context to be
    # checked against instead of an empty buffer - without this, the first
    # turn after character creation had nothing to check against at all.
    if narration.get("narrative"):
        player.append_narrative(
            str(narration["narrative"]),
            summary=narration.get("scene_summary"),
        )
        try:
            extracted = extract_mechanical_deltas(player_state, str(narration["narrative"]))
            player.apply_extracted_deltas(extracted)
        except Exception as exc:
            print(f"Note: mechanical extraction failed for opening scene (game continues normally): {exc}")
        save_store.save_playthrough(save_id, player.to_save_dict())

    return jsonify(
        {
            "player_state": player_state,
            "narration": narration,
            "bio": bio,
            "save_id": save_id,
        }
    )


@app.route("/api/new-game/stream", methods=["POST"])
def new_game_stream():
    """
    Streaming counterpart to /api/new-game: identical character-creation
    logic (_build_new_character is shared), but the two prose-generating
    stages - the bio and the opening scene - are streamed to the client as
    Server-Sent Events as they're generated, rather than the client seeing
    nothing at all until the entire multi-call chain (character gen, bio,
    opening scene planning, opening scene writing) finishes.

    Character generation itself is NOT streamed - it uses JSON mode
    (response_format=json_object), which cannot produce valid partial JSON
    mid-stream, the same reason plan_scene isn't token-streamed either.
    It's a real, sometimes-slow blocking call; the "preparing" event covers
    that wait the same way it already does for a normal turn's
    classification stage.

    Event sequence:
      event: preparing           data: {}                       (fires immediately)
      [character generation runs here if mode is random/custom - a blocking
       JSON call, no event emitted for it specifically]
      event: character_ready     data: {"player_state": ...}     (fires once character + session/save are set up)
      event: bio_chunk           data: {"text": "..."}           (repeated, as the bio streams)
      [plan_scene runs here for the opening scene - blocking, no event emitted]
      event: narrative_chunk     data: {"text": "..."}           (repeated, as the opening scene streams)
      event: final               data: {full response, same shape /api/new-game returns}
      event: error                data: {"error": "..."}         (only on failure)
    """
    data = request.get_json(silent=True) or {}

    # session_id and save_id are generated and written to Flask's `session`
    # HERE, in the view function's own body, not inside generate() below.
    # This is the same fix already required twice elsewhere in this file
    # (see _finalize_turn's docstring and perform_action_stream's
    # session_id_for_bg comment): `session` is a Flask request-context-local
    # object, safe to touch in a view function's own body but NOT safe from
    # inside a generator Werkzeug iterates after this function has already
    # returned. game_sessions/session_save_ids, by contrast, are plain
    # module-level dicts with no such restriction - those writes stay
    # inside generate(), only the Flask `session` writes had to move.
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    save_id = str(uuid.uuid4())
    session["save_id"] = save_id

    def sse_event(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"

    def generate():
        yield sse_event("preparing", {})

        try:
            player = _build_new_character(data)
        except Exception as exc:
            yield sse_event("error", {"error": f"Character creation failed: {exc}"})
            return

        game_sessions[session_id] = player
        session_save_ids[session_id] = save_id
        save_store.save_playthrough(save_id, player.to_save_dict())

        player_state = player.export_engine_state()

        yield sse_event("character_ready", {"player_state": player_state, "save_id": save_id})

        bio_chunks = []
        try:
            for chunk in stream_bio(player_state):
                bio_chunks.append(chunk)
                yield sse_event("bio_chunk", {"text": chunk})
            bio = "".join(bio_chunks).strip() or (player_state.get("background") or "No background available.")
        except Exception:
            # Same fallback generate_bio itself uses - never leave the
            # summary screen blank just because this call failed.
            bio = player_state.get("background") or "No background available."

        initial_action = "Character creation complete. Introduce the character's starting situation."
        try:
            plan = plan_scene(player_state, initial_action, {"dc": 0, "roll": 0, "outcome": "start"})
        except Exception as exc:
            narration = {
                "narrative": f"Welcome to the life simulation! (Fallback scene: {exc})",
                "choices": ["Look around.", "Check your pocket.", "Find a job."],
                "state_deltas": {},
            }
            yield sse_event("narrative_chunk", {"text": narration["narrative"]})
        else:
            narrative_chunks = []
            try:
                for chunk in write_scene(plan):
                    narrative_chunks.append(chunk)
                    yield sse_event("narrative_chunk", {"text": chunk})
                narrative_text = "".join(narrative_chunks)
                narrative_text = truncate_scene_if_needed(narrative_text, plan.get("target_words", 550))
                narration = {
                    "narrative": narrative_text,
                    "choices": plan["choices"],
                    "state_deltas": plan["state_deltas"],
                    "scene_summary": plan.get("scene_summary", ""),
                }
            except Exception as exc:
                narration = {
                    "narrative": f"Welcome to the life simulation! (Fallback scene: {exc})",
                    "choices": ["Look around.", "Check your pocket.", "Find a job."],
                    "state_deltas": {},
                }
                yield sse_event("narrative_chunk", {"text": narration["narrative"]})

        # Seed the rolling narrative buffer with the opening scene itself,
        # same as the non-streaming route does - without this, the
        # player's very first real action has nothing to check continuity
        # against.
        if narration.get("narrative"):
            player.append_narrative(
                str(narration["narrative"]),
                summary=narration.get("scene_summary"),
            )
            try:
                extracted = extract_mechanical_deltas(player_state, str(narration["narrative"]))
                player.apply_extracted_deltas(extracted)
            except Exception as exc:
                print(f"Note: mechanical extraction failed for opening scene (game continues normally): {exc}")
            save_store.save_playthrough(save_id, player.to_save_dict())

        final_player_state = player.export_engine_state()

        yield sse_event("final", {
            "player_state": final_player_state,
            "narration": narration,
            "bio": bio,
            "save_id": save_id,
        })

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def _prepare_turn(player, action_text: str, locked_outcome):
    """
    Shared pre-narration setup for a turn: time-skip detection/advancement,
    superuser command parsing, action classification, and dice resolution.
    Everything up through "we now know what happened mechanically, and just
    need the model to narrate it" - identical whether the eventual
    narration call is the old single-shot JSON version or the new
    streaming version, so this is extracted once rather than duplicated
    across two routes. Returns a dict with everything narration needs:
    player_state, superuser, classification, dice_outcome, time_skip_info.

    Mutates player in place (advance_date, process_payday, etc.) exactly
    as the original inline code did - callers should treat player as
    already updated by the time this returns.
    """
    time_skip_info = None
    bare_time_skip = False
    skip_match = parse_time_skip(action_text)
    if skip_match:
        unit, amount, action_text = skip_match
        old_date, new_date = player.advance_date(unit, amount)
        time_skip_info = {
            "unit": unit,
            "amount": amount,
            "from": old_date.strftime("%m/%d/%Y"),
            "to": new_date.strftime("%m/%d/%Y"),
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

        births = player.check_pregnancies()
        if births:
            time_skip_info["births"] = births

        stale_threads = player.check_stale_case_file()
        if stale_threads:
            time_skip_info["stale_threads"] = stale_threads

        recovered_injuries = player.check_injury_recovery()
        if recovered_injuries:
            time_skip_info["recovered_injuries"] = recovered_injuries

        bare_time_skip = not action_text
        if bare_time_skip:
            action_text = "Time passes."

    player_state = player.export_engine_state()

    superuser = parse_superuser_command(action_text)

    if superuser:
        classification = {
            "stat_used": None,
            "dc_input": None,
            "e_modifier": 0,
            "reasoning": "Superuser command - classification bypassed.",
        }

        if superuser["type"] == "question":
            dice_outcome = {"outcome": "superuser_question", "superuser_override": True}
        else:
            if superuser["type"] == "dice_override":
                dice_outcome = {"outcome": superuser["outcome"], "superuser_override": True}
            else:
                dice_outcome = {"outcome": "superuser_command", "superuser_override": True}

    else:
        if bare_time_skip:
            classification = {
                "stat_used": "charisma",
                "dc_input": "auto",
                "e_modifier": 0,
                "reasoning": "Bare time skip, no action to classify.",
                "elapsed_unit": time_skip_info["unit"] if time_skip_info else "h",
                "elapsed_amount": time_skip_info["amount"] if time_skip_info else 1,
            }
        else:
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

        if time_skip_info is None:
            est_unit = classification.get("elapsed_unit", "h")
            est_amount = classification.get("elapsed_amount", 1)
            # force_ignore_same_scene fires once the model has already
            # been given _SAME_SCENE_GRACE_TURNS worth of turns to
            # naturally wrap up a scene after a pending pace change and
            # kept reporting same_scene_continuation=true anyway (tracked
            # by _finalize_turn from PRIOR turns, checked here against
            # THIS turn's floor call) - see _apply_sim_speed_floor's own
            # docstring for why prompt instructions alone were not
            # sufficient here.
            force_ignore = bool(
                player.pending_speed_change
                and player.same_scene_streak_since_pace_change >= _SAME_SCENE_GRACE_TURNS
            )
            est_unit, est_amount = _apply_sim_speed_floor(
                est_unit, est_amount,
                player_state.get("sim_speed", "d"),
                same_scene_continuation=bool(classification.get("same_scene_continuation")),
                force_ignore_same_scene=force_ignore,
            )
            if force_ignore:
                # The forced jump happened this turn - the pending change
                # has now genuinely been applied, nothing left to enforce.
                print(f"[pace-change diagnostic] FORCED transition: model still reported "
                      f"same_scene_continuation={classification.get('same_scene_continuation')!r} but "
                      f"the floor was applied anyway (est now {est_unit}={est_amount}).")
                player.pending_speed_change = None
                player.same_scene_streak_since_pace_change = 0
            try:
                old_hour = player.hour
                old_date, new_date = player.advance_date(est_unit, est_amount)

                clock_advanced = (old_date != new_date) or (player.hour != old_hour)
                if clock_advanced:
                    paychecks = player.process_payday()
                    expense_payments = player.process_expenses()
                    drifted_npcs = player.check_npc_drift()
                    births = player.check_pregnancies()
                    stale_threads = player.check_stale_case_file()
                    recovered_injuries = player.check_injury_recovery()

                    results = {
                        "paychecks": paychecks,
                        "expense_payments": expense_payments,
                        "drifted_npcs": drifted_npcs,
                        "births": births,
                        "stale_threads": stale_threads,
                        "recovered_injuries": recovered_injuries,
                    }
                    notable = any(results.values())

                    if old_date != new_date or notable:
                        time_skip_info = {
                            "unit": est_unit,
                            "amount": est_amount,
                            "from": old_date.strftime("%m/%d/%Y"),
                            "to": new_date.strftime("%m/%d/%Y"),
                            "estimated": True,
                        }
                        if old_date == new_date:
                            time_skip_info["same_day"] = True
                        for field_name, value in results.items():
                            if value:
                                time_skip_info[field_name] = value

                    player_state = player.export_engine_state()
            except ValueError:
                pass

        stat_name = classification.get("stat_used") or "charisma"
        stat_value = getattr(player, stat_name, None)
        if stat_value is None:
            stat_value = player.charisma

        total_e_modifier = (
            classification.get("e_modifier", 0)
            + player.gear_modifier_for(stat_name)
            - player.injury_penalty_for(stat_name)
        )

        dc_input = classification.get("dc_input", "med_risk")
        if is_no_check(dc_input):
            dice_outcome = auto_success_outcome()
            if locked_outcome:
                dice_outcome["override_skipped"] = True
                dice_outcome["override_skipped_reason"] = (
                    "This action was routine and never rolled a check, so the "
                    "requested dice override had nothing to apply to."
                )
        elif locked_outcome:
            dice_outcome = {"outcome": locked_outcome, "superuser_override": True, "locked": True}
        else:
            try:
                dice_outcome = resolve_check(
                    stat_value,
                    dc_input,
                    total_e_modifier,
                )
            except Exception as exc:
                dice_outcome = {
                    "dc": 23,
                    "roll": 0,
                    "outcome": f"error: {exc}",
                }

    return {
        "action_text": action_text,
        "player_state": player_state,
        "superuser": superuser,
        "classification": classification,
        "dice_outcome": dice_outcome,
        "time_skip_info": time_skip_info,
    }


def _finalize_turn(player, superuser, narration, classification, dice_outcome, prepared_player_state, action_text: str, session_id: str | None, save_id: str | None):
    """
    Shared post-narration handling: choice-cadence enforcement, applying
    state deltas, appending to the narrative buffer, kicking off background
    mechanical extraction, auto-saving, and building the final response
    payload. Shared between the non-streaming route and the streaming
    route's final SSE event so the two paths can never silently diverge in
    what counts as a "real" turn.

    session_id and save_id are passed in as plain strings rather than read
    from Flask's `session` object inside this function, on purpose: this
    function can be called from inside perform_action_stream's generate()
    generator, which Werkzeug iterates as the response streams out - which
    can happen after this function's caller has already returned and the
    request context has torn down. Reading session.get(...) (directly, or
    indirectly via get_current_save_id(), which also reads session
    internally) at that point raises "Working outside of request context" -
    a real crash seen twice in production logs, once for each of these two
    values, before both were made explicit parameters. Every caller must
    capture both from its own view function body, while request context is
    still guaranteed active, and pass them in here - never read `session`
    directly, and never call get_current_save_id(), from inside this
    function.
    """
    is_ooc_turn = bool(
        superuser and superuser.get("type") in ("question", "command")
    )

    # Streak tracking only - see Player.same_scene_streak_since_pace_change's
    # own comment for the full mechanism. This function only updates the
    # count; the actual decision to force the pace change happens in
    # _prepare_turn, which checks this same streak BEFORE this turn's own
    # classification is known, then clears pending_speed_change once the
    # force actually fires (see _prepare_turn for where that clearing
    # happens - not here, since only _prepare_turn knows whether this
    # turn's floor call was actually forced). A pure question turn is
    # exempt (it never runs classification with real narrative content,
    # so its same_scene_continuation value is meaningless here).
    if player.pending_speed_change and not (superuser and superuser.get("type") == "question"):
        if classification.get("same_scene_continuation"):
            player.same_scene_streak_since_pace_change += 1
        else:
            # The model genuinely moved to a new scene on its own -
            # nothing left to force, the transition already happened
            # naturally.
            player.pending_speed_change = None
            player.same_scene_streak_since_pace_change = 0

    if not is_ooc_turn:
        has_real_choices = isinstance(narration.get("choices"), list) and len(narration["choices"]) > 0
        choice_enforcement_debug = {
            "counter_before_this_turn": player.consecutive_empty_choice_turns,
            "had_real_choices_this_turn": has_real_choices,
            "forcing_attempted": False,
            "forcing_succeeded": None,
        }

        if has_real_choices:
            player.consecutive_empty_choice_turns = 0
        else:
            # No separate staleness check needed here - get_current_player
            # now validates the whole player object against SQLite's
            # turn_number on every request (see its docstring), so by the
            # time this function runs, `player` (and therefore this
            # counter) is already guaranteed current, not just this one
            # field in isolation.
            player.consecutive_empty_choice_turns += 1

            if player.consecutive_empty_choice_turns >= 2:
                choice_enforcement_debug["forcing_attempted"] = True
                forced_choices = None
                for _attempt in range(2):
                    try:
                        candidate = plan_scene(
                            prepared_player_state,
                            action_text,
                            dice_outcome,
                            force_choices=True,
                        )
                    except Exception:
                        continue

                    if isinstance(candidate.get("choices"), list) and len(candidate["choices"]) > 0:
                        forced_choices = candidate
                        break

                if forced_choices is not None:
                    narration["choices"] = forced_choices["choices"]
                    if isinstance(forced_choices.get("state_deltas"), dict):
                        narration.setdefault("state_deltas", {})
                        narration["state_deltas"].update(forced_choices["state_deltas"])
                    player.consecutive_empty_choice_turns = 0
                    choice_enforcement_debug["forcing_succeeded"] = True
                else:
                    narration["narrative"] = (
                        str(narration.get("narrative", "")).rstrip()
                        + "\n\n(The story is having trouble presenting a real "
                        + "choice here. Try describing what you want to do next.)"
                    )
                    narration["choices"] = []
                    player.consecutive_empty_choice_turns = 0
                    choice_enforcement_debug["forcing_succeeded"] = False
    else:
        choice_enforcement_debug = None

    deltas = narration.get("state_deltas", {})
    if isinstance(deltas, dict):
        player.apply_deltas(deltas)

    player.turn_number += 1

    if narration.get("narrative") and not is_ooc_turn:
        player.append_narrative(
            str(narration["narrative"]),
            summary=narration.get("scene_summary"),
        )

        narrative_text = str(narration["narrative"])
        thread = threading.Thread(
            target=_apply_extraction_async,
            args=(player, session_id, save_id, prepared_player_state, narrative_text),
            daemon=True,
        )
        thread.start()

    if save_id:
        try:
            save_store.save_playthrough(save_id, player.to_save_dict())
        except Exception as exc:
            print(f"Auto-save failed (game continues normally): {exc}")

    final_player_state = player.export_engine_state()

    return {
        "classification": classification,
        "dice_outcome": dice_outcome,
        "narration": narration,
        "player_state": final_player_state,
        "_choice_enforcement_debug": choice_enforcement_debug,
    }


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

    locked_outcome_label = data.get("locked_outcome")
    locked_outcome = None
    if locked_outcome_label:
        normalized = str(locked_outcome_label).strip().lower()
        locked_outcome = DICE_OVERRIDE_OUTCOMES.get(normalized)

    try:
        prepared = _prepare_turn(player, action_text, locked_outcome)
    except ValueError as exc:
        return jsonify({"error": f"Invalid time skip: {exc}"}), 400

    player_state = prepared["player_state"]
    superuser = prepared["superuser"]
    classification = prepared["classification"]
    dice_outcome = prepared["dice_outcome"]
    time_skip_info = prepared["time_skip_info"]
    action_text = prepared["action_text"]

    if superuser and superuser["type"] == "question":
        try:
            answer = answer_question(player_state, superuser["raw"])
        except Exception as exc:
            answer = f"[Could not answer: {exc}]"
        narration = {"narrative": answer, "choices": [], "state_deltas": {}}
    elif superuser:
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
        try:
            narration = narrate_outcome(player_state, action_text, dice_outcome, time_skip=time_skip_info)
        except Exception as exc:
            narration = {
                "narrative": f"Action complete. (Narration error: {exc})",
                "choices": ["Continue", "Look around"],
                "state_deltas": {},
            }

    result = _finalize_turn(player, superuser, narration, classification, dice_outcome, player_state, action_text, session.get("session_id"), get_current_save_id())
    return jsonify(result)


@app.route("/api/action/stream", methods=["POST"])
def perform_action_stream():
    """
    Streaming counterpart to /api/action: identical turn resolution
    (_prepare_turn is shared), but the narrative prose is streamed to the
    client as Server-Sent Events as it's generated, rather than the client
    waiting for one full JSON response. Event sequence:
      event: preparing         data: {}                    (fires immediately, before classification/dice resolution)
      event: dice_outcome      data: {"dice_outcome": ...}  (fires once, right after _prepare_turn resolves)
      [plan_scene runs here - a blocking call, no event emitted; scene_outline,
       era_constraints, choices, and state_deltas are all decided before any
       prose exists]
      event: narrative_chunk   data: {"text": "..."}        (repeated, as write_scene streams prose from the locked plan)
      event: final             data: {full response, same shape /api/action returns}
      event: error             data: {"error": "..."}       (only on failure)
    A pure ?...? question and a superuser command still narrate in one
    shot rather than streaming (neither writes long-form prose, so there's
    nothing to gain from streaming them, and it keeps this endpoint's
    special-case handling limited to the one path that actually benefits).
    """
    player = get_current_player()
    if not player:
        return jsonify({"error": "No active game session found. Please start a new game."}), 400

    data = request.get_json(silent=True) or {}
    action_text = data.get("action", "").strip()

    if not action_text:
        return jsonify({"error": "Action cannot be empty."}), 400

    locked_outcome_label = data.get("locked_outcome")
    locked_outcome = None
    if locked_outcome_label:
        normalized = str(locked_outcome_label).strip().lower()
        locked_outcome = DICE_OVERRIDE_OUTCOMES.get(normalized)

    # Captured here, not inside _finalize_turn, and not inside generate()
    # below. `session` is a Flask request-context-local object - safe to
    # read here (perform_action_stream's own body, guaranteed inside an
    # active request), but NOT safe to read later from inside generate(),
    # which Werkzeug iterates as the response streams out and which can
    # keep running after this view function has already returned and the
    # request context has torn down. Reading session.get() from inside the
    # generator crashed with "Working outside of request context" in
    # production - this plain string is captured up front instead, while
    # it's still guaranteed safe, and threaded through as a parameter from
    # here on. save_id has the exact same requirement: get_current_save_id()
    # also reads `session` directly, and hit the identical crash the first
    # time this was overlooked for it specifically.
    session_id_for_bg = session.get("session_id")
    save_id_for_bg = get_current_save_id()

    def sse_event(event_name: str, payload: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"

    def generate():
        # _prepare_turn (classification + dice resolution) runs INSIDE the
        # generator, not before it, and the very first thing this yields is
        # a "preparing" event with no real payload. This matters more than
        # it looks: Flask does not actually open the HTTP response stream
        # to the client until the generator's first yield happens, so if
        # _prepare_turn ran before generate() was even called (as it used
        # to), the player's connection sat completely idle for the full
        # duration of the classification call - a real, separate LLM round
        # trip - before a single byte of the "streaming" response existed.
        # That defeated the entire point of streaming: time to first byte
        # was still gated behind a full extra model call. Yielding
        # immediately opens the connection right away; classification then
        # runs while the connection is already live, so the player at least
        # sees the request acknowledged instead of nothing at all until
        # narration starts.
        yield sse_event("preparing", {})

        turn_start = time.monotonic()

        try:
            prepared = _prepare_turn(player, action_text, locked_outcome)
        except ValueError as exc:
            yield sse_event("error", {"error": f"Invalid time skip: {exc}"})
            return

        classification_done_at = time.monotonic()

        player_state = prepared["player_state"]
        superuser = prepared["superuser"]
        classification = prepared["classification"]
        dice_outcome = prepared["dice_outcome"]
        time_skip_info = prepared["time_skip_info"]
        resolved_action_text = prepared["action_text"]

        # dice_outcome AND the turn's resolved date/header are both already
        # known from _prepare_turn, well before any narration call starts -
        # emit both immediately so the frontend can update the dice badge
        # and the top-bar date right away instead of waiting for the whole
        # streamed scene (plus the choices/deltas follow-up call after it)
        # to finish. Without this, the displayed date silently lagged one
        # full turn behind Python's actual tracked date for the entire
        # duration of streaming - the correct date was already known, it
        # just wasn't being sent until the very end.
        yield sse_event("dice_outcome", {
            "dice_outcome": dice_outcome,
            "header": player_state.get("header"),
            "location": player_state.get("location"),
        })

        if superuser and superuser["type"] == "question":
            try:
                answer = answer_question(player_state, superuser["raw"])
            except Exception as exc:
                answer = f"[Could not answer: {exc}]"
            narration = {"narrative": answer, "choices": [], "state_deltas": {}}
            yield sse_event("narrative_chunk", {"text": answer})
            result = _finalize_turn(player, superuser, narration, classification, dice_outcome, player_state, resolved_action_text, session_id_for_bg, save_id_for_bg)
            yield sse_event("final", result)
            return

        if superuser:
            try:
                narration = narrate_outcome(
                    player_state, resolved_action_text, dice_outcome,
                    superuser_command=superuser["raw"], time_skip=time_skip_info,
                )
            except Exception as exc:
                narration = {
                    "narrative": f"[Superuser command received, but narration failed: {exc}]",
                    "choices": ["Continue"],
                    "state_deltas": {},
                }
            yield sse_event("narrative_chunk", {"text": narration.get("narrative", "")})
            result = _finalize_turn(player, superuser, narration, classification, dice_outcome, player_state, resolved_action_text, session_id_for_bg, save_id_for_bg)
            yield sse_event("final", result)
            return

        # Planning happens BEFORE any prose is streamed - this is the real
        # architectural shift from the old design (stream prose first, then
        # a follow-up call for choices/deltas after). plan_scene decides
        # scene_outline, era_constraints, choices, and state_deltas all at
        # once, using DeepSeek V4 Pro's reasoning pass; write_scene then
        # turns the locked scene_outline into prose with zero judgment
        # calls of its own. This costs more visible latency before the
        # first narrative_chunk arrives (planning is a full blocking call,
        # not hidden behind reading time the way the old follow-up call
        # was) - an accepted, deliberate tradeoff for continuity/era
        # accuracy, not an oversight. The "preparing" event already covers
        # this wait on the frontend, same as it always has.
        try:
            plan = plan_scene(player_state, resolved_action_text, dice_outcome, time_skip=time_skip_info)
        except Exception as exc:
            yield sse_event("error", {"error": f"Scene planning failed: {exc}"})
            return

        planning_done_at = time.monotonic()
        # No other event naturally lands here (dice_outcome fires before
        # planning starts, narrative_chunk fires once prose starts) - this
        # is the one real gap in the pipeline with nothing marking it, and
        # it's also the single most expensive stage (DeepSeek V4 Pro's
        # reasoning call), so it gets its own event specifically to make
        # "which stage is this turn stuck in" answerable from the browser
        # console alone, without needing server log access.
        yield sse_event("stage_timing", {
            "classification_seconds": round(classification_done_at - turn_start, 2),
            "planning_seconds": round(planning_done_at - classification_done_at, 2),
        })

        narrative_chunks = []
        first_chunk_at = None
        try:
            for chunk in write_scene(plan):
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                narrative_chunks.append(chunk)
                yield sse_event("narrative_chunk", {"text": chunk})
        except Exception as exc:
            yield sse_event("error", {"error": f"Narration streaming failed: {exc}"})
            return

        writing_done_at = time.monotonic()

        narrative_text = "".join(narrative_chunks)
        # Server-side truncation safety net (see truncate_scene_if_needed's
        # docstring) - if this trims the text, the streamed chunks the
        # player already saw will be longer than this final value. That's
        # intentional and already handled: the frontend's "final" event
        # handler already reconciles its displayed text against
        # narration.narrative whenever they differ (originally built for
        # the forced-choices retry rewriting narrative text) - the same
        # mechanism trims the display down to match here, with no frontend
        # changes needed.
        narrative_text = truncate_scene_if_needed(narrative_text, plan.get("target_words", 550))

        narration = {
            "narrative": narrative_text,
            "choices": plan["choices"],
            "state_deltas": plan["state_deltas"],
            "scene_summary": plan.get("scene_summary", ""),
        }

        result = _finalize_turn(player, superuser, narration, classification, dice_outcome, player_state, resolved_action_text, session_id_for_bg, save_id_for_bg)
        result["_stage_timing"] = {
            "classification_seconds": round(classification_done_at - turn_start, 2),
            "planning_seconds": round(planning_done_at - classification_done_at, 2),
            "time_to_first_prose_token_seconds": round((first_chunk_at - planning_done_at), 2) if first_chunk_at else None,
            "writing_seconds": round(writing_done_at - planning_done_at, 2),
            "total_seconds": round(writing_done_at - turn_start, 2),
        }
        yield sse_event("final", result)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


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
        session["save_id"] = requested_save_id
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

    # Only flagged as a pending change if the pace is genuinely different
    # from what it already was - re-tapping the same speed (or the
    # frontend re-sending the current value) should not force-close
    # whatever scene is currently in progress for no reason.
    if speed != player.sim_speed:
        player.pending_speed_change = speed
        player.same_scene_streak_since_pace_change = 0

    player.sim_speed = speed

    # Bumped even though this isn't a narrated turn - get_current_player's
    # multi-worker staleness guard (see its own docstring) only reloads a
    # worker's cached Player from SQLite when the database's turn_number is
    # AHEAD of what that worker has in memory. Without bumping it here, a
    # speed change made on worker A was invisible to worker B: SQLite had
    # the new sim_speed, but worker B's cache had no signal anything had
    # changed, so it kept serving its own stale copy indefinitely - a real,
    # confirmed bug (speed changes silently not applying to the next turn
    # if it happened to land on a different worker than the one that set
    # the speed). Any write that should be visible across workers needs to
    # move this counter, not just narrated turns specifically.
    player.turn_number += 1

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
