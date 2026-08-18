from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple, Optional, ClassVar
import random

try:
    from dateutil.relativedelta import relativedelta
except ImportError as exc:
    raise ImportError(
        "python-dateutil is required for calendar-correct date advancement "
        "(pip3.12 install --user python-dateutil)"
    ) from exc


def _clamped_stat(raw_value, default: int = 10) -> int:
    """
    Parses and clamps a stat value to the valid 0-20 range when restoring a
    Player from a save file. A save is another untrusted input path (hand
    edited, or corrupted), so this mirrors the same clamp apply_deltas already
    enforces for in-game changes, and the one app.py enforces at creation.
    """
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0, min(20, value))


def _migrate_narrative_scenes(data: dict) -> list:
    """
    Reconstructs narrative_scenes when loading a save file, handling both
    the current format and saves written before the two-tier narrative
    memory system existed. Current-format saves already have a
    narrative_scenes list and are used as-is. Older saves only have the
    legacy flat last_narrative string (scenes concatenated with '---'
    separators) - those are split back into a scene list on a best-effort
    basis, since '---' was already the real scene boundary the old flat
    buffer used, just never stored as a structured list. This intentionally
    does NOT try to backfill scene_summaries for an old save's earlier
    history - that history was never at full text with a real summary
    written for it, and inventing one now (or leaving it silently absent)
    is a smaller loss than trying to fabricate summaries for scenes this
    code never actually saw.
    """
    if "narrative_scenes" in data:
        return data.get("narrative_scenes") or []

    legacy_blob = data.get("last_narrative", "")
    if not legacy_blob:
        return []

    scenes = [s.strip() for s in legacy_blob.split("---")]
    return [s for s in scenes if s]


@dataclass
class Player:
    # Personal Info & Calendar Tracker
    name: str = "Unknown"
    age: int = 18
    date: date = field(default_factory=lambda: date(2024, 1, 1))
    hour: int = 8  # 0-23, clock time within the current date; see advance_date for how &h moves this
    location: str = "New York, NY"
    race: str = ""

    # Core stats (D&D scale: 0 to 20)
    health: int = 10
    max_health: int = 20  # permanently reduced by severe injuries; health can never exceed this
    strength: int = 10
    charisma: int = 10
    intelligence: int = 10
    willpower: int = 10
    stress: int = 0

    # Structured as: [{"description": "Lost use of left hand in a factory accident", "health_cap_reduction": 4, "date": "03/15/2031"}]
    permanent_injuries: List[Dict[str, Any]] = field(default_factory=list)

    cash: float = 0.0  # Liquid spending money ($)
    
    # Character context
    occupation: str = ""
    background: str = ""

    # Job & income loop
    job_title: str = ""
    salary: float = 0.0  # amount paid per pay_frequency
    pay_frequency: str = "biweekly"  # one of: weekly, biweekly, monthly
    last_paycheck_date: date = field(default_factory=lambda: date(2024, 1, 1))

    # Recurring expenses loop (rent, bills, debt payments, etc.)
    # Structured as: [{"id": "rent", "name": "Rent", "amount": 650.0, "frequency": "monthly", "last_paid_date": "2024-01-01"}]
    expenses: List[Dict[str, Any]] = field(default_factory=list)

    # NPC background-agency loop: tracks when we last checked whether any
    # relationship should drift on its own, independent of the player visiting.
    last_npc_drift_date: date = field(default_factory=lambda: date(2024, 1, 1))

    # Difficulty/tone preset chosen at character creation: "gritty", "standard", or "forgiving"
    difficulty_preset: str = "standard"

    # Genre chosen once at character creation: "realism", "fantasy", or "horror".
    # Fixed for the life of this character - no delta key exists to change it
    # mid-playthrough; switching genres means starting a new life.
    genre: str = "realism"

    # Two-tier narrative memory, replacing the old flat word-capped string.
    # narrative_scenes holds the most recent RECENT_SCENES_FULL_TEXT_COUNT
    # scenes at FULL fidelity - exact prose, nothing lost - so the model can
    # check a new scene against what literally just happened (including a
    # couple of turns back, not only the single most recent one) to catch
    # same-scene contradictions and keep tone/detail consistent across a
    # short run of turns. Ordered oldest -> newest; use append_narrative to
    # add to it, never assign directly.
    #
    # scene_summaries holds everything OLDER than that: each entry is a
    # short, model-written compression of a scene that has aged out of full
    # text, not deleted outright. This is the actual fix for continuity
    # drift over a longer span - rather than one giant flat buffer where
    # every scene (recent or 40 turns old) competes equally for the model's
    # attention at full raw-text weight, only the truly recent scenes get
    # that weight; everything older is pre-digested into a sentence or two,
    # which is both cheaper per scene AND easier for the model to actually
    # use, since a long list of short summaries is far less noisy per fact
    # than an equivalent length of undifferentiated prose. Ordered oldest ->
    # newest; capped by SCENE_SUMMARY_MAX_COUNT (oldest summaries eventually
    # drop off entirely - the goal is a long, USEFUL memory span, not
    # unbounded storage of every scene forever).
    narrative_scenes: List[str] = field(default_factory=list)
    scene_summaries: List[Dict[str, str]] = field(default_factory=list)

    # A simple monotonic counter, incremented once per real player turn
    # (see app.py). Exists solely so save_store.save_playthrough can refuse
    # to overwrite a save with a LOWER turn_number than what's already on
    # disk - without this, two saves racing (the main turn's synchronous
    # save, and the background extraction thread's follow-up save a moment
    # later, or two overlapping requests on a multi-worker deployment) had
    # no way to tell which one was actually newer, so a slow write could
    # silently clobber a fast one and visibly roll the date backward. This
    # is a correctness guard, not narrative content - never surfaced to the
    # model or the player.
    turn_number: int = 0

    # Tracks consecutive turns where narrate_outcome returned an empty
    # choices list. Exists so app.py can deterministically force a real
    # choice set rather than leaving "does this turn get choices" entirely
    # up to the model's own judgment call - the model still writes the
    # choices themselves (Python can't invent a meaningful narrative fork
    # out of nothing), but it does NOT get to decide, on its own, to go
    # more than one turn in a row without offering any. Reset to 0 the
    # moment a turn returns real choices; incremented on empty ones. Never
    # surfaced to the model or the player - same category as turn_number.
    consecutive_empty_choice_turns: int = 0

    # Active pregnancies, tracked so a birth is a real, calendar-driven event
    # rather than something the model has to remember and narrate correctly
    # unaided nine months later. Structured as:
    # [{"partner": "Elena Ruiz", "conception_date": "2031-03-15", "due_date": "2031-12-15", "child_name": null}]
    pregnancies: List[Dict[str, Any]] = field(default_factory=list)

    # Gear/equipment that grants a mechanical bonus on relevant stat checks,
    # distinct from the flavor-only `inventory` list below. Structured as:
    # [{"name": "Lockpick set", "e_modifier": 2, "applies_to": "intelligence"}]
    gear: List[Dict[str, Any]] = field(default_factory=list)

    # Temporary injuries with a real recovery date, distinct from
    # permanent_injuries above which never heal. Structured as:
    # [{"description": "Sprained wrist", "stat": "strength", "penalty": 2, "recovery_date": "2026-03-01"}]
    recovering_injuries: List[Dict[str, Any]] = field(default_factory=list)


    # Persistent simulation pacing: one of "h", "d", "w", "m", "y". This is the
    # ongoing default the model biases toward every turn until explicitly
    # changed again, distinct from a one-time &-skip which jumps forward once
    # and does not alter this setting.
    sim_speed: str = "d"

    # Set the moment sim_speed actually changes (see app.py's set_speed
    # route). Earlier versions of this mechanism tried a purely prompt-
    # based approach - first a one-shot hard-cut instruction, then a
    # softer two-stage "conclude this turn, switch next turn" version -
    # and real production logs showed BOTH were correctly delivered to
    # the model (confirmed via direct prompt/response logging) and simply
    # not complied with: the model kept narrating the same hour-scale
    # scene turn after turn regardless of what it was told. This is the
    # enforcement version: pending_speed_change holds the target speed
    # (None when no change is pending). same_scene_streak_since_pace_change
    # counts consecutive turns, since the change, where the model's own
    # classifier still reported same_scene_continuation=true despite being
    # told a pace change was pending - once this exceeds the allowed
    # grace period (see app.py's _apply_sim_speed_floor), Python forces
    # the time-skip directly rather than asking again.
    pending_speed_change: str | None = None
    same_scene_streak_since_pace_change: int = 0

    # Persistent simulation state
    skills: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    reputation: Dict[str, int] = field(default_factory=dict)
    debt: float = 0.0
    status_flags: List[str] = field(default_factory=list)

    # Detailed History & Social Tracking (matching visual reference)
    # Structured as: {"Person Name": {"relation": "Mother", "quality": 85, "status": "Living"}}
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Standing facts about the world - places, objects, and details that
    # aren't about a specific person and so don't belong in relationships,
    # but still need to survive past both narrative memory tiers below
    # (narrative_scenes/scene_summaries) - a scene-level summary compresses
    # what HAPPENED, not standalone facts like a diner's name or a scar's
    # origin that were only ever mentioned in passing. Without this, such a
    # detail is simply gone once its originating scene ages out of full text
    # and never made it into a summary - nothing else in the engine
    # remembers it. Structured as a flat dict of short key -> one-sentence
    # fact, e.g. {"Rosa's Diner": "The family restaurant on Elm Street, run
    # by her father until it closed in 1998."}. Kept intentionally simple
    # (no tags/dates/tiers like relationships have) since this is meant to
    # be cheap reference material, not another subsystem with its own
    # lifecycle to manage.
    world_facts: Dict[str, str] = field(default_factory=dict)

    # Structured as: [{"year": 2024, "age": 18, "event": "Graduated High School", "impact": "Positive"}]
    life_events: List[Dict[str, Any]] = field(default_factory=list)

    # Curated durable memory, survives regardless of how long life_events grows.
    # Structured as: [{"summary": "Shot two crew members during a robbery", "status": "open", "tags": ["crime"]}]
    case_file: List[Dict[str, Any]] = field(default_factory=list)

    def adjust_cash(self, amount: float) -> float:
        """
        The one sanctioned way to change self.cash by a relative amount.
        cash is a float, and floats cannot represent most cents exactly
        (0.1 + 0.2 != 0.3 in binary floating point) - across hundreds of
        turns of paychecks, expenses, business resolutions, and trades, raw
        += / -= drifts into visible garbage like $3502.4399999999996.
        Rounding to the cent on every single mutation, right here, stops
        that error from ever compounding. Returns the new cash value for
        convenience at call sites that want it.
        """
        self.cash = round(self.cash + amount, 2)
        return self.cash

    # How many of the most recent scenes stay at full text. 3 is generously
    # beyond a single turn - enough for the model to catch same-scene
    # contradictions and keep tone consistent across a short run of turns,
    # without the noise of an ever-growing flat buffer. Scenes older than
    # this move to scene_summaries instead of being dropped outright.
    RECENT_SCENES_FULL_TEXT_COUNT: ClassVar[int] = 3

    # How many scene summaries to retain once a scene ages out of full
    # text. Summaries are cheap (a sentence or two each), so this can cover
    # far more real history than raw text ever could at comparable cost -
    # the point isn't unbounded storage, it's a long, genuinely USEFUL
    # memory span rather than 2500 words of undifferentiated recent prose.
    SCENE_SUMMARY_MAX_COUNT: ClassVar[int] = 40

    @property
    def last_narrative(self) -> str:
        """
        Read-only compatibility view: the full text of narrative_scenes,
        joined the same way the old flat buffer used to look, for any code
        (extraction, save-file readers, etc.) that still wants "the recent
        narrative" as one string. Does NOT include scene_summaries - callers
        that need the older, summarized history should read scene_summaries
        directly rather than expect it flattened in here, since mixing full
        text and summaries into one blob is exactly the undifferentiated-
        noise problem this two-tier structure exists to avoid.
        """
        return "\n\n---\n\n".join(self.narrative_scenes)

    def append_narrative(self, scene_text: str, summary: str | None = None) -> None:
        """
        The one sanctioned way to add a scene to narrative memory. Appends
        scene_text to narrative_scenes (full fidelity); if that now exceeds
        RECENT_SCENES_FULL_TEXT_COUNT, the OLDEST scene is popped out of
        full text. If summary was provided (the model writes one every turn
        as part of its normal structured output - see
        generate_structured_followup's scene_summary field), that summary
        is what gets pushed into scene_summaries as the popped scene's
        compressed representation; if no summary was given (e.g. an older
        save from before this existed, or the model omitted it), a short
        mechanical fallback derived from the raw text is used instead,
        rather than silently losing that scene from memory entirely once it
        ages out of full text.
        """
        scene_text = (scene_text or "").strip()
        if not scene_text:
            return

        self.narrative_scenes.append(scene_text)

        if len(self.narrative_scenes) > self.RECENT_SCENES_FULL_TEXT_COUNT:
            aged_out_scene = self.narrative_scenes.pop(0)

            fallback_summary = " ".join(aged_out_scene.split()[:40])
            if len(aged_out_scene.split()) > 40:
                fallback_summary += "..."

            self.scene_summaries.append({
                "summary": (summary or "").strip() or fallback_summary,
                "date": self.formatted_date(),
            })

            if len(self.scene_summaries) > self.SCENE_SUMMARY_MAX_COUNT:
                self.scene_summaries = self.scene_summaries[-self.SCENE_SUMMARY_MAX_COUNT:]

    def advance_date(self, unit: str, amount: int) -> Tuple[date, date]:
        """
        Advances self.date (and, for 'h', self.hour) by amount units (unit is
        one of: 'y', 'm', 'w', 'd', 'h'). Python is the sole authority for this
        calculation; the model is never asked to compute or invent a new date
        or time, only told the result.

        Uses relativedelta for years/months so calendar edge cases resolve
        correctly (e.g. Jan 31 + 1 month lands on Feb 28/29, not an invalid
        date or a naive +30 days that silently drifts across month boundaries).

        'h' (hours) advances self.hour and rolls over into self.date via
        divmod once it crosses 24, so a big hour skip (e.g. &20h) still lands
        on the correct date and clock time, not silently discarded.

        Any larger-than-hour skip (d/w/m/y) resets self.hour to 8 (a sensible
        start-of-day default) rather than leaving a stale clock time sitting
        there from whatever hour an earlier &h left it at - a multi-day skip
        shouldn't claim to land at, say, 11 PM on the new date for no reason.

        Returns (old_date, new_date) so the caller can report exactly what
        changed; callers that also care about the clock time should read
        self.hour directly before/after this call.
        """
        old_date = self.date
        if unit == "y":
            self.date = self.date + relativedelta(years=amount)
            self.hour = 8
        elif unit == "m":
            self.date = self.date + relativedelta(months=amount)
            self.hour = 8
        elif unit == "w":
            self.date = self.date + timedelta(weeks=amount)
            self.hour = 8
        elif unit == "d":
            self.date = self.date + timedelta(days=amount)
            self.hour = 8
        elif unit == "h":
            total_hours = self.hour + amount
            days_to_add, new_hour = divmod(total_hours, 24)
            self.hour = new_hour
            if days_to_add:
                self.date = self.date + timedelta(days=days_to_add)
        else:
            raise ValueError(f"Unknown time unit: {unit!r} (expected one of y/m/w/d/h)")

        # Age advances automatically alongside the calendar rather than being
        # a separate thing the model has to remember to update.
        self.age = self._age_for_date(self.date, old_date)

        return old_date, self.date

    def formatted_time(self) -> str:
        """12-hour clock format with AM/PM, e.g. '2:00 PM', '8:00 AM'."""
        period = "AM" if self.hour < 12 else "PM"
        display_hour = self.hour % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:00 {period}"

    def process_payday(self) -> List[Dict[str, Any]]:
        """
        Checks whether one or more pay periods have elapsed since last_paycheck_date
        and adds cash for each one that has. Returns a list of paycheck events
        (empty if none occurred), so the caller can report them to the player.

        Multiple periods can fire in one call if a large time skip (e.g. &6m)
        jumped past several paydays at once - each one pays out individually
        rather than being silently skipped or merged into one lump sum, since
        real paychecks don't work that way.
        """
        if self.salary <= 0 or not self.job_title:
            return []

        period_days = {"weekly": 7, "biweekly": 14, "monthly": 30}.get(self.pay_frequency, 14)
        payouts = []

        # Guard against a runaway loop if last_paycheck_date is somehow far in
        # the future or malformed; cap at 60 payouts per call (covers roughly
        # 2+ years of weekly pay in one jump, more than any single time skip
        # should realistically produce).
        safety_limit = 60
        while (self.date - self.last_paycheck_date).days >= period_days and safety_limit > 0:
            self.last_paycheck_date = self.last_paycheck_date + timedelta(days=period_days)
            self.adjust_cash(self.salary)
            payouts.append({
                "date": self.last_paycheck_date.strftime("%m/%d/%Y"),
                "amount": self.salary,
                "job_title": self.job_title,
            })
            safety_limit -= 1

        return payouts

    def add_expense(self, name: str, amount: float, frequency: str = "monthly") -> None:
        """
        Adds a new recurring expense (rent, a car payment, a debt installment,
        etc.). Starts its own last_paid_date at today, so the first deduction
        lands one full period out rather than firing immediately or never.
        Multiple expenses can coexist and are tracked independently.
        """
        expense_id = f"{name.lower().replace(' ', '_')}_{len(self.expenses)}"
        self.expenses.append({
            "id": expense_id,
            "name": name,
            "amount": amount,
            "frequency": frequency if frequency in ("weekly", "biweekly", "monthly") else "monthly",
            "last_paid_date": self.date.isoformat(),
        })

    def remove_expense(self, name: str) -> bool:
        """
        Removes the first expense matching name (case-insensitive), e.g. when
        the player pays off a debt or moves out of an apartment. Returns True
        if something was actually removed.
        """
        target = name.strip().lower()
        for i, expense in enumerate(self.expenses):
            if expense.get("name", "").strip().lower() == target:
                self.expenses.pop(i)
                return True
        return False

    def process_expenses(self) -> List[Dict[str, Any]]:
        """
        Symmetrical to process_payday: checks every recurring expense and
        deducts cash for each period that has elapsed since it was last paid.
        Returns a list of payment events for the caller to report. Cash is
        allowed to go negative here (debt/overdraft), matching how paychecks
        can push it arbitrarily high; the story is expected to react to a
        negative balance rather than Python silently blocking the deduction.
        """
        period_days_map = {"weekly": 7, "biweekly": 14, "monthly": 30}
        all_payments = []

        for expense in self.expenses:
            period_days = period_days_map.get(expense.get("frequency", "monthly"), 30)
            try:
                last_paid = date.fromisoformat(expense.get("last_paid_date", self.date.isoformat()))
            except (ValueError, TypeError):
                last_paid = self.date

            safety_limit = 60
            while (self.date - last_paid).days >= period_days and safety_limit > 0:
                last_paid = last_paid + timedelta(days=period_days)
                amount = expense.get("amount", 0.0)
                self.adjust_cash(-(amount))
                all_payments.append({
                    "date": last_paid.strftime("%m/%d/%Y"),
                    "amount": amount,
                    "name": expense.get("name", "Expense"),
                })
                safety_limit -= 1

            expense["last_paid_date"] = last_paid.isoformat()

        return all_payments

    def normalize_relationship_scales(self) -> bool:
        """
        One-time repair for saves created before quality/reputation were
        clamped to 0-20. Values that leaked out of range (e.g. quality: 35
        from unconstrained model output) get pulled back into range rather
        than staying broken until the next unrelated delta happens to touch
        that specific NPC or group. Returns True if anything was actually
        changed, so the caller can decide whether to re-save.
        """
        changed = False
        for entry in self.relationships.values():
            if isinstance(entry, dict) and "quality" in entry:
                try:
                    clamped = max(0, min(20, int(entry["quality"])))
                except (TypeError, ValueError):
                    clamped = 10
                if clamped != entry["quality"]:
                    entry["quality"] = clamped
                    changed = True
        for group_name, value in list(self.reputation.items()):
            try:
                clamped = max(0, min(20, int(value)))
            except (TypeError, ValueError):
                clamped = 10
            if clamped != value:
                self.reputation[group_name] = clamped
                changed = True
        return changed

    def add_gear(self, name: str, e_modifier: int, applies_to: str) -> Dict[str, Any]:
        item = {"name": name, "e_modifier": e_modifier, "applies_to": applies_to}
        self.gear.append(item)
        return item

    def remove_gear(self, name: str) -> bool:
        for i, item in enumerate(self.gear):
            if item["name"] == name:
                del self.gear[i]
                return True
        return False

    def gear_modifier_for(self, stat_name: str) -> int:
        """
        Total e_modifier bonus from all carried gear applicable to a given
        stat check. Called by app.py when resolving a check, added on top
        of whatever e_modifier the model itself assigned - a real,
        Python-guaranteed bonus the model doesn't have to remember to apply.
        """
        return sum(item.get("e_modifier", 0) for item in self.gear if item.get("applies_to") == stat_name)

    # --- Temporary injury recovery ---------------------------------------------

    def add_recovering_injury(self, description: str, stat: str, penalty: int, recovery_days: int = 21) -> Dict[str, Any]:
        record = {
            "description": description,
            "stat": stat,
            "penalty": penalty,
            "recovery_date": (self.date + timedelta(days=recovery_days)).isoformat(),
        }
        self.recovering_injuries.append(record)
        return record

    def injury_penalty_for(self, stat_name: str) -> int:
        """Total temporary penalty currently active against a given stat, from all unresolved injuries."""
        return sum(inj.get("penalty", 0) for inj in self.recovering_injuries if inj.get("stat") == stat_name)

    def check_injury_recovery(self) -> List[Dict[str, Any]]:
        """Fires at time-skip resolution: clears any injury whose recovery_date has passed."""
        recovered = []
        still_recovering = []
        for inj in self.recovering_injuries:
            try:
                recovery_date = date.fromisoformat(inj["recovery_date"])
            except (KeyError, ValueError):
                still_recovering.append(inj)
                continue
            if self.date >= recovery_date:
                recovered.append(inj)
            else:
                still_recovering.append(inj)
        self.recovering_injuries = still_recovering
        return recovered

    # --- Heat / bust resolution ---------------------------------------------

    def start_pregnancy(self, partner: str = "") -> Dict[str, Any]:
        """
        Begins tracking a new pregnancy from today's in-fiction date. Due date
        is set 40 weeks (280 days) out, matching standard human gestation.
        Python owns this calculation the same way it owns dice rolls and
        paydays; the model only ever supplies who the partner is, never the
        due date itself. Returns the created pregnancy record.
        """
        record = {
            "partner": partner or "Unknown",
            "conception_date": self.date.isoformat(),
            "due_date": (self.date + timedelta(days=280)).isoformat(),
            "child_name": None,
        }
        self.pregnancies.append(record)
        return record

    def check_pregnancies(self) -> List[Dict[str, Any]]:
        """
        Symmetrical to check_npc_drift/check_heat_all_contexts: fires after
        every calendar advance and resolves any pregnancy whose due_date has
        been reached or passed. A birth is no longer something the model has
        to remember and narrate consistently on its own months later - once
        conceived, Python guarantees it actually happens on schedule.

        Each resolved pregnancy is removed from the active list, logged to
        life_events (category Family), and added to relationships as a new
        NPC (quality 15, a warm default for a newborn) so the child persists
        in state going forward. Returns the list of birth events that fired
        this call, empty if none were due yet.
        """
        births = []
        still_pregnant = []
        for record in self.pregnancies:
            try:
                due = date.fromisoformat(record.get("due_date", ""))
            except (TypeError, ValueError):
                still_pregnant.append(record)
                continue

            if self.date >= due:
                child_name = record.get("child_name") or "the baby"
                partner = record.get("partner", "Unknown")
                self.relationships[child_name] = {
                    "relation": "Child",
                    "quality": 15,
                    "status": "Living",
                }
                self.log_life_event(
                    event=f"Gave birth (with {partner})" if partner != "Unknown" else "Gave birth",
                    impact="Positive",
                    category="Family",
                )
                births.append({
                    "partner": partner,
                    "child_name": child_name,
                    "date": self.formatted_date(),
                })
            else:
                still_pregnant.append(record)

        self.pregnancies = still_pregnant
        return births

    def check_npc_drift(self, threshold_days: int = 30, drift_chance: float = 0.35) -> List[str]:
        """
        Fires after time skips, independent of whether the player actually
        visited anyone. If at least threshold_days have passed since the last
        check, rolls a per-NPC chance (drift_chance) for each named relationship
        to be flagged as due for a background life update this cycle, then
        resets the tracker. Returns the names of NPCs selected to drift, empty
        if the threshold hasn't been reached or nobody was selected.

        Python owns the randomness here the same way it owns dice rolls; the
        model is only ever told which NPCs were selected, never asked to
        decide who or how often on its own.
        """
        if (self.date - self.last_npc_drift_date).days < threshold_days:
            return []

        self.last_npc_drift_date = self.date

        selected = [
            name for name in self.relationships
            if random.random() < drift_chance
        ]
        return selected

    def _age_for_date(self, current: date, previous_reference: date) -> int:
        """
        Recomputes age from how many full years have passed since the game's
        original starting date is not tracked separately, so this approximates
        by adding whole years elapsed since the last date to the existing age.
        Good enough for a life-sim's purposes; not meant to model exact
        birthdays unless the narrative tracks one explicitly via life_events.
        """
        years_elapsed = relativedelta(current, previous_reference).years
        return self.age + years_elapsed

    def formatted_date(self) -> str:
        """MM/DD/YYYY, matching the header format already used in the UI/prompt."""
        return self.date.strftime("%m/%d/%Y")

    LIFE_EVENT_CATEGORIES = {
        "Crime", "Career", "Relationship", "Family", "Health",
        "Legal", "Financial", "Education", "Other",
    }

    def log_life_event(self, event: str, impact: str = "Neutral", category: str = "Other") -> None:
        """Helper to append a structured event to the player's life history timeline."""
        if category not in self.LIFE_EVENT_CATEGORIES:
            category = "Other"
        self.life_events.append({
            "year": self.date.year,
            "month": self.date.month,
            "age": self.age,
            "event": event,
            "impact": impact,
            "category": category,
        })

    def add_permanent_injury(self, description: str, health_cap_reduction: int) -> None:
        """
        Permanently lowers max_health by health_cap_reduction (clamped so it
        can never drop below 1, since 0 would make the character unable to
        ever have any health at all). If current health is now above the new
        lower cap, it's pulled down to match immediately rather than sitting
        in an impossible above-cap state until the next unrelated health delta.
        Also logs the injury to life_events (category Health) automatically,
        so a lasting injury always leaves a record even if the model forgets
        to log it separately.
        """
        if health_cap_reduction <= 0 or not description:
            return
        self.permanent_injuries.append({
            "description": description,
            "health_cap_reduction": health_cap_reduction,
            "date": self.formatted_date(),
        })
        self.max_health = max(1, self.max_health - health_cap_reduction)
        if self.health > self.max_health:
            self.health = self.max_health
        self.log_life_event(event=description, impact="Negative", category="Health")

    def update_case_file(self, updates: List[Dict[str, Any]]) -> None:
        """
        Applies additions/resolutions to the durable case file. Each update is either
        a new entry ({"summary": ..., "tags": [...]}) or a resolution
        ({"resolve": "text that matches an existing summary"}).
        Caps at 20 open entries, dropping the oldest resolved ones first to make room.
        """
        if not isinstance(updates, list):
            return
        for update in updates:
            if not isinstance(update, dict):
                continue
            if "resolve" in update:
                target = update["resolve"]
                for entry in self.case_file:
                    if entry.get("summary") == target and entry.get("status") == "open":
                        entry["status"] = "resolved"
                        break
            elif update.get("summary"):
                self.case_file.append({
                    "summary": update["summary"],
                    "status": "open",
                    "tags": update.get("tags", []),
                    "created_date": self.date.isoformat(),
                })
        open_count = sum(1 for e in self.case_file if e.get("status") == "open")
        if open_count > 20:
            self.case_file = [e for e in self.case_file if e.get("status") == "open"][-20:]
        elif len(self.case_file) > 30:
            resolved = [e for e in self.case_file if e.get("status") == "resolved"]
            open_entries = [e for e in self.case_file if e.get("status") == "open"]
            keep_resolved = max(0, 30 - len(open_entries))
            self.case_file = open_entries + resolved[-keep_resolved:] if keep_resolved else open_entries

    def check_stale_case_file(self, staleness_days: int = 90) -> List[Dict[str, Any]]:
        """
        Symmetrical to check_npc_drift/check_pregnancies: fires after every
        calendar advance. Selecting which threads have gone stale is NOT
        left to the model's judgment - a prompt-only "consider surfacing
        this" instruction is a suggestion, and suggestions are exactly what
        this project has repeatedly found the model quietly skips under
        real load (word count, dice-rolling, crew extraction all had this
        same failure mode before they were moved to a hard Python check).
        Python decides which entries are stale by simple date math; the
        model is only ever told which ones were selected, as a fact it must
        act on this turn, never as an option to weigh.

        An entry with no created_date (older saves, before this field
        existed) is treated as stale immediately - safer to surface an old
        thread once than to let a truly ancient one go unaddressed forever
        because it predates the field that tracks its age.
        """
        stale = []
        for entry in self.case_file:
            if entry.get("status") != "open":
                continue
            created_str = entry.get("created_date")
            if not created_str:
                stale.append(entry)
                continue
            try:
                created = date.fromisoformat(created_str)
            except ValueError:
                stale.append(entry)
                continue
            if (self.date - created).days >= staleness_days:
                stale.append(entry)
        return stale

    # Keyword -> (applies_to stat, default e_modifier) for auto-promoting an
    # inventory item into real Gear instead of letting it sit as inert flavor
    # text. Deliberately conservative (+1 only) - a small guaranteed effect
    # is much better than a plausible weapon/tool doing literally nothing
    # because the model happened to phrase it into inventory instead of
    # calling add_gear, but this should not hand out a large bonus on its
    # own; a bigger, narratively-justified bonus should still come from an
    # explicit add_gear delta. Checked as substring match, case-insensitive.
    _GEAR_AUTO_PROMOTE_KEYWORDS: ClassVar[Dict[str, Tuple[str, int]]] = {
        "lockpick": ("intelligence", 1), "pick set": ("intelligence", 1),
        "knife": ("strength", 1), "gun": ("strength", 1), "pistol": ("strength", 1),
        "rifle": ("strength", 1), "blade": ("strength", 1), "weapon": ("strength", 1),
        "brass knuckles": ("strength", 1), "bat": ("strength", 1),
        "disguise": ("charisma", 1), "fake id": ("charisma", 1), "forged": ("charisma", 1),
        "toolkit": ("intelligence", 1), "tool kit": ("intelligence", 1),
        "crowbar": ("strength", 1),
    }

    def _route_inventory_item(self, item: str) -> None:
        """
        Called for every item about to be added to plain flavor-only
        inventory. If the item's name matches an obvious mechanical-use
        keyword (a weapon, lockpicks, a disguise, tools), it's redirected
        into real Gear via add_gear instead - so a weapon narrated straight
        into inventory doesn't silently do nothing on a future check, even
        if the model didn't think to call add_gear itself. Anything that
        doesn't match stays exactly as it is today: flavor-only inventory.
        """
        lowered = item.lower()
        for keyword, (applies_to, modifier) in self._GEAR_AUTO_PROMOTE_KEYWORDS.items():
            if keyword in lowered:
                if not any(g["name"] == item for g in self.gear):
                    self.add_gear(name=item, e_modifier=modifier, applies_to=applies_to)
                return
        if item not in self.inventory:
            self.inventory.append(item)

    def apply_deltas(self, state_deltas: Dict[str, Any]) -> None:
        """
        Applies state updates returned by narrate_outcome to the player object.
        Supports stat increments/decrements, list additions/removals, dict updates, and event logging.
        """
        if not isinstance(state_deltas, dict):
            return
        for key, delta in state_deltas.items():
            # max_health is never set directly via a raw delta - it only moves
            # through add_permanent_injury below, so every change to it has an
            # actual reason attached rather than an unexplained number.
            if key == "max_health":
                continue
            # 1. Update Numeric Attributes
            if hasattr(self, key):
                current_val = getattr(self, key)    
                # Direct numeric adjustments (e.g. "health": -2, "cash": 50.0, "stress": 3)
                if isinstance(current_val, (int, float)) and isinstance(delta, (int, float)):
                    new_val = current_val + delta
                    # Health is capped by max_health, not a flat 20 - a permanent
                    # injury can lower this ceiling, so health can never fully
                    # recover past whatever the character's current cap is.
                    if key == "health":
                        new_val = max(0, min(self.max_health, int(new_val)))
                    # Clamp the other core D&D stats between 0 and 20
                    elif key in {"strength", "charisma", "intelligence", "willpower"}:
                        new_val = max(0, min(20, int(new_val)))
                    # Clamp Stress between 0 and 20
                    elif key == "stress":
                        new_val = max(0, min(20, int(new_val)))
                    # Cash is real money and must round to the cent on every
                    # single mutation, not just at display time - binary
                    # floats can't represent most cents exactly (0.1 + 0.2 !=
                    # 0.3), so repeated +=/-= across hundreds of turns drifts
                    # into visible garbage like $3502.4399999999996 if it's
                    # never snapped back to two decimal places. Rounding here,
                    # at the single place all narrator-driven cash deltas
                    # apply, stops the error before it can compound.
                    elif key == "cash":
                        new_val = round(new_val, 2)
                    setattr(self, key, new_val)
                # 2. Append to Lists (e.g. "inventory": ["Lockpick"], "status_flags": ["Wounded"])
                elif key == "inventory" and isinstance(current_val, list) and isinstance(delta, list):
                    # Special-cased ahead of the generic list-append below:
                    # every incoming inventory item is checked for an obvious
                    # mechanical use first, rather than always landing as
                    # inert flavor text.
                    for item in delta:
                        self._route_inventory_item(item)
                elif isinstance(current_val, list) and isinstance(delta, list):
                    for item in delta:
                        if item not in current_val:
                            current_val.append(item)
                # 3. Update Dictionary Mappings (e.g. "skills", "relationships", "reputation")
                elif isinstance(current_val, dict) and isinstance(delta, dict):
                    if key == "relationships":
                        # Deep-merge each NPC's own dict rather than replacing it wholesale,
                        # so a delta that only touches e.g. status_note doesn't silently
                        # wipe out relation/quality/status set in an earlier turn.
                        for npc_name, npc_delta in delta.items():
                            if isinstance(npc_delta, dict) and isinstance(current_val.get(npc_name), dict):
                                current_val[npc_name].update(npc_delta)
                            else:
                                current_val[npc_name] = npc_delta
                            # quality is a 0-20 scale, same range as the core stats
                            # (10 = neutral starting point). Clamp here rather than
                            # trusting the model to stay in range, since nothing
                            # previously enforced this and values like 35 leaked
                            # through unbounded.
                            entry = current_val.get(npc_name)
                            if isinstance(entry, dict) and "quality" in entry:
                                try:
                                    entry["quality"] = max(0, min(20, int(entry["quality"])))
                                except (TypeError, ValueError):
                                    entry["quality"] = 10
                    elif key == "reputation":
                        current_val.update(delta)
                        # Same 0-20 scale as relationship quality, applied to every
                        # group/faction entry that was just touched by this delta.
                        for group_name in delta:
                            if group_name in current_val:
                                try:
                                    current_val[group_name] = max(0, min(20, int(current_val[group_name])))
                                except (TypeError, ValueError):
                                    current_val[group_name] = 10
                    else:
                        current_val.update(delta)
                # 3b. Overwrite plain string fields (e.g. "job_title", "pay_frequency",
                # "occupation"). Starting a new job resets last_paycheck_date to today
                # so the first payday lands one full pay period from now, rather than
                # firing immediately (if left at the old job's date) or never (if it
                # stayed far in the past).
                elif isinstance(current_val, str) and isinstance(delta, str):
                    if key == "job_title" and delta != current_val:
                        self.last_paycheck_date = self.date
                    setattr(self, key, delta)
            # 4. Handle Item Removals (e.g. "remove_inventory": ["Crowbar"])
            if key == "remove_inventory" and isinstance(delta, list):
                for item in delta:
                    if item in self.inventory:
                        self.inventory.remove(item)
            # 5. Handle Status Flag Removals (e.g. "remove_status": ["Wounded"])
            if key == "remove_status" and isinstance(delta, list):
                for item in delta:
                    if item in self.status_flags:
                        self.status_flags.remove(item)
            # 6. Log New Life Event if provided in deltas
            # e.g. "add_life_event": {"event": "Arrested for burglary", "impact": "Negative", "category": "Crime"}
            if key == "add_life_event" and isinstance(delta, dict):
                self.log_life_event(
                    event=delta.get("event", "Unknown Event"),
                    impact=delta.get("impact", "Neutral"),
                    category=delta.get("category", "Other"),
                )
            # 6b. Log a PERMANENT injury - lowers max_health going forward, not
            # just a temporary health hit that can fully heal back to 20.
            # e.g. "add_permanent_injury": {"description": "Lost two fingers in a press accident", "health_cap_reduction": 3}
            if key == "add_permanent_injury" and isinstance(delta, dict):
                try:
                    reduction = int(delta.get("health_cap_reduction", 0))
                except (TypeError, ValueError):
                    reduction = 0
                self.add_permanent_injury(
                    description=delta.get("description", "").strip(),
                    health_cap_reduction=reduction,
                )
            # 6c. Shift the persistent simulation pace (e.g. tightening to Daily
            # after a severe event). This changes the ongoing default, not a
            # one-time skip - it stays in effect until changed again.
            # e.g. "set_sim_speed": "d"
            if key == "set_sim_speed" and isinstance(delta, str) and delta in ("h", "d", "w", "m", "y"):
                self.sim_speed = delta
            # 7. Update the durable case file
            # e.g. "case_file_updates": [{"summary": "Killed two crew members in a robbery", "tags": ["crime"]}]
            if key == "case_file_updates" and isinstance(delta, list):
                self.update_case_file(delta)
            # 8. Add a new recurring expense
            # e.g. "add_expense": {"name": "Rent", "amount": 650.0, "frequency": "monthly"}
            if key == "add_expense" and isinstance(delta, dict) and delta.get("name"):
                self.add_expense(
                    name=delta["name"],
                    amount=float(delta.get("amount", 0.0)),
                    frequency=delta.get("frequency", "monthly"),
                )
            # 9. Remove a recurring expense by name
            # e.g. "remove_expense": "Rent"
            if key == "remove_expense" and isinstance(delta, str) and delta.strip():
                self.remove_expense(delta)
            # 18. Begin tracking a new pregnancy. Python computes the due date;
            # the model only supplies the partner's name. The birth itself
            # fires automatically via check_pregnancies() once due_date is
            # reached, so it never depends on the model remembering to
            # narrate it correctly months later.
            # e.g. "start_pregnancy": {"partner": "Elena Ruiz"}
            if key == "start_pregnancy" and isinstance(delta, dict):
                self.start_pregnancy(partner=delta.get("partner", ""))
            # 23. Acquire gear that grants a mechanical bonus on a specific
            # stat check going forward (a lockpick set, a weapon, a disguise).
            # e.g. "add_gear": {"name": "Lockpick set", "e_modifier": 2, "applies_to": "intelligence"}
            if key == "add_gear" and isinstance(delta, dict) and delta.get("name"):
                self.add_gear(
                    name=delta["name"],
                    e_modifier=int(delta.get("e_modifier", 0)),
                    applies_to=delta.get("applies_to", "charisma"),
                )
            # 26. Remove gear by name (lost, sold, confiscated, broken).
            # e.g. "remove_gear": "Lockpick set"
            if key == "remove_gear" and isinstance(delta, str) and delta.strip():
                self.remove_gear(delta)
            # 27. Add a temporary injury with a real recovery window, distinct
            # from add_permanent_injury which never heals.
            # e.g. "add_recovering_injury": {"description": "Sprained wrist", "stat": "strength", "penalty": 2, "recovery_days": 21}
            if key == "add_recovering_injury" and isinstance(delta, dict) and delta.get("description"):
                self.add_recovering_injury(
                    description=delta["description"],
                    stat=delta.get("stat", "strength"),
                    penalty=int(delta.get("penalty", 1)),
                    recovery_days=int(delta.get("recovery_days", 21)),
                )

    def apply_extracted_deltas(self, extracted: Dict[str, Any]) -> None:
        """
        Consumes the output of Narrator.extract_mechanical_deltas - the
        dedicated, narrow pass that reads a scene's prose and reports
        structured mechanical events (a pregnancy, a new expense, gear
        acquired, an injury, a new relationship). This is the ONLY path any
        of these events reach the engine through; narrate_outcome's own
        state_deltas doesn't carry any of these keys at all (see
        SYSTEM_PROMPT section 5 in Narrator.py).

        Every value coming out of extraction is still just a model's
        judgment call, so nothing here trusts it blindly: every numeric
        field is coerced with a safe default via .get()/float()/int(), every
        call that can fail on bad input is wrapped in try/except, and a
        failure in one category never prevents the rest from applying.
        Unknown/missing categories are simply skipped - most turns will have
        few or none of these fire, which is expected and correct.
        """
        if not isinstance(extracted, dict):
            return

        pregnancy = extracted.get("new_pregnancy")
        if isinstance(pregnancy, dict):
            self.start_pregnancy(partner=pregnancy.get("partner", ""))

        expense = extracted.get("new_expense")
        if isinstance(expense, dict) and expense.get("name"):
            try:
                self.add_expense(
                    name=expense["name"],
                    amount=float(expense.get("amount", 0.0)),
                    frequency=expense.get("frequency", "monthly"),
                )
            except (ValueError, TypeError) as exc:
                print(f"Note: extracted new_expense failed - {exc}")

        removed_expense = extracted.get("removed_expense")
        if isinstance(removed_expense, str) and removed_expense.strip():
            self.remove_expense(removed_expense)

        gear = extracted.get("new_gear")
        if isinstance(gear, dict) and gear.get("name"):
            try:
                self.add_gear(
                    name=gear["name"],
                    e_modifier=int(gear.get("e_modifier", 0)),
                    applies_to=gear.get("applies_to", "charisma"),
                )
            except (ValueError, TypeError) as exc:
                print(f"Note: extracted new_gear failed - {exc}")

        removed_gear = extracted.get("removed_gear")
        if isinstance(removed_gear, str) and removed_gear.strip():
            self.remove_gear(removed_gear)

        rec_injury = extracted.get("new_recovering_injury")
        if isinstance(rec_injury, dict) and rec_injury.get("description"):
            try:
                self.add_recovering_injury(
                    description=rec_injury["description"],
                    stat=rec_injury.get("stat", "strength"),
                    penalty=int(rec_injury.get("penalty", 1)),
                    recovery_days=int(rec_injury.get("recovery_days", 21)),
                )
            except (ValueError, TypeError) as exc:
                print(f"Note: extracted new_recovering_injury failed - {exc}")

        perm_injury = extracted.get("new_permanent_injury")
        if isinstance(perm_injury, dict) and perm_injury.get("description"):
            try:
                self.add_permanent_injury(
                    description=perm_injury["description"],
                    health_cap_reduction=int(perm_injury.get("health_cap_reduction", 0)),
                )
            except (ValueError, TypeError) as exc:
                print(f"Note: extracted new_permanent_injury failed - {exc}")

        new_rels = extracted.get("new_relationships")
        if isinstance(new_rels, dict) and new_rels:
            # Route through the same apply_deltas relationships branch so
            # quality clamping (0-20) and the deep-merge-vs-new-NPC logic
            # are identical to every other relationships write path -
            # there is only one place that logic lives, this just calls it.
            self.apply_deltas({"relationships": new_rels})

        new_facts = extracted.get("new_world_facts")
        if isinstance(new_facts, dict) and new_facts:
            # world_facts is a flat str->str dict, so the generic dict-merge
            # branch in apply_deltas already handles it correctly (a plain
            # .update(), no clamping needed like relationships/reputation) -
            # still routed through apply_deltas rather than set directly
            # here, so there's one single place any state_deltas-shaped
            # write ever happens, same reasoning as new_relationships above.
            self.apply_deltas({"world_facts": new_facts})

    def export_engine_state(self) -> Dict[str, Any]:
        """
        Packages the player state into a dictionary for internal tracking or
        LLM payloads. Deliberately does NOT include current_time/hour - the
        AI only ever sees the date, never the time of day. self.hour is
        still fully tracked internally (advance_date needs it for correct
        midnight-rollover math on hour-scale skips), it just never leaves
        Python. Reasoning: time-of-day was adding a second, finer-grained
        axis the narrator had to keep consistent on top of the date, and it
        wasn't buying enough narrative value to be worth that surface -
        the date alone is enough for the story to reference "this morning"
        or "that evening" in prose without Python needing to track and
        enforce an exact clock hour through the model.
        """
        return {
            "header": f"{self.name} | Age: {self.age} | Date: {self.formatted_date()}",
            "age": self.age,
            "current_date": self.formatted_date(),
            "location": self.location,
            "race": self.race,
            "stats": {
                "health": self.health,
                "strength": self.strength,
                "charisma": self.charisma,
                "intelligence": self.intelligence,
                "willpower": self.willpower,
                "stress": self.stress,
            },
            "max_health": self.max_health,
            "permanent_injuries": self.permanent_injuries,
            "cash": self.cash,
            "occupation": self.occupation,
            "background": self.background,
            "job_title": self.job_title,
            "salary": self.salary,
            "pay_frequency": self.pay_frequency,
            "expenses": self.expenses,
            "difficulty_preset": self.difficulty_preset,
            "genre": self.genre,
            "recent_scenes_full": self.narrative_scenes,
            "earlier_scenes_summary": self.scene_summaries,
            "relationships": self.relationships,
            "world_facts": self.world_facts,
            "reputation": self.reputation,
            "debt": self.debt,
            "status_flags": self.status_flags,
            "life_events": self.life_events,
            "case_file": [e for e in self.case_file if e.get("status") == "open"],
            "sim_speed": self.sim_speed,
            "pending_speed_change": self.pending_speed_change,
            "same_scene_streak_since_pace_change": self.same_scene_streak_since_pace_change,
            "inventory": self.inventory,
            "gear": self.gear,
            "skills": self.skills,
            "pregnancies": self.pregnancies,
            "recovering_injuries": self.recovering_injuries,
        }

    def to_save_dict(self) -> Dict[str, Any]:
        """
        Full raw snapshot of every field, for save-game export. Unlike
        export_engine_state (which is trimmed for what the LLM needs each turn,
        e.g. only open case_file entries), this keeps everything, including
        resolved case_file entries, so a reload is a true continuation.
        """
        return {
            "name": self.name,
            "age": self.age,
            "date": self.date.isoformat(),
            "hour": self.hour,
            "location": self.location,
            "race": self.race,
            "health": self.health,
            "max_health": self.max_health,
            "permanent_injuries": self.permanent_injuries,
            "strength": self.strength,
            "charisma": self.charisma,
            "intelligence": self.intelligence,
            "willpower": self.willpower,
            "stress": self.stress,
            "cash": self.cash,
            "occupation": self.occupation,
            "background": self.background,
            "job_title": self.job_title,
            "salary": self.salary,
            "pay_frequency": self.pay_frequency,
            "last_paycheck_date": self.last_paycheck_date.isoformat(),
            "expenses": self.expenses,
            "last_npc_drift_date": self.last_npc_drift_date.isoformat(),
            "difficulty_preset": self.difficulty_preset,
            "genre": self.genre,
            "narrative_scenes": self.narrative_scenes,
            "scene_summaries": self.scene_summaries,
            "turn_number": self.turn_number,
            "consecutive_empty_choice_turns": self.consecutive_empty_choice_turns,
            "pregnancies": self.pregnancies,
            "gear": self.gear,
            "recovering_injuries": self.recovering_injuries,
            "sim_speed": self.sim_speed,
            "pending_speed_change": self.pending_speed_change,
            "same_scene_streak_since_pace_change": self.same_scene_streak_since_pace_change,
            "skills": self.skills,
            "inventory": self.inventory,
            "reputation": self.reputation,
            "debt": self.debt,
            "status_flags": self.status_flags,
            "relationships": self.relationships,
            "world_facts": self.world_facts,
            "life_events": self.life_events,
            "case_file": self.case_file,
        }

    @classmethod
    def from_save_dict(cls, data: Dict[str, Any]) -> "Player":
        """Rebuilds a Player from a to_save_dict() snapshot. Missing/malformed
        keys fall back to the normal dataclass defaults rather than raising."""
        if not isinstance(data, dict):
            data = {}

        raw_date = data.get("date")
        if raw_date:
            parsed_date = date.fromisoformat(raw_date)
        elif data.get("year"):
            # Backward compatibility: saves created before the calendar rework
            # only had month (a name string) + year (an int). Best-effort
            # convert to the 1st of that month rather than losing the save.
            month_names = [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
            ]
            month_str = str(data.get("month", "January")).strip().lower()
            month_num = month_names.index(month_str) + 1 if month_str in month_names else 1
            parsed_date = date(int(data["year"]), month_num, 1)
        else:
            parsed_date = date(2024, 1, 1)

        raw_paycheck_date = data.get("last_paycheck_date")
        parsed_paycheck_date = date.fromisoformat(raw_paycheck_date) if raw_paycheck_date else parsed_date

        raw_drift_date = data.get("last_npc_drift_date")
        parsed_drift_date = date.fromisoformat(raw_drift_date) if raw_drift_date else parsed_date

        try:
            parsed_hour = int(data.get("hour", 8))
        except (TypeError, ValueError):
            parsed_hour = 8
        parsed_hour = max(0, min(23, parsed_hour))

        try:
            parsed_max_health = int(data.get("max_health", 20))
        except (TypeError, ValueError):
            parsed_max_health = 20
        parsed_max_health = max(1, min(20, parsed_max_health))

        player = cls(
            name=data.get("name", "Unknown"),
            age=int(data.get("age", 18)),
            date=parsed_date,
            hour=parsed_hour,
            location=data.get("location", "New York, NY"),
            race=data.get("race", ""),
            health=_clamped_stat(data.get("health", 10)),
            max_health=parsed_max_health,
            strength=_clamped_stat(data.get("strength", 10)),
            charisma=_clamped_stat(data.get("charisma", 10)),
            intelligence=_clamped_stat(data.get("intelligence", 10)),
            willpower=_clamped_stat(data.get("willpower", 10)),
            stress=_clamped_stat(data.get("stress", 0), default=0),
            cash=float(data.get("cash", 0.0)),
            occupation=data.get("occupation", ""),
            background=data.get("background", ""),
            debt=float(data.get("debt", 0.0)),
            job_title=data.get("job_title", ""),
            salary=float(data.get("salary", 0.0)),
            pay_frequency=data.get("pay_frequency", "biweekly"),
            last_paycheck_date=parsed_paycheck_date,
            last_npc_drift_date=parsed_drift_date,
            difficulty_preset=data.get("difficulty_preset", "standard"),
            genre=data.get("genre") if data.get("genre") in ("realism", "fantasy", "horror") else "realism",
            narrative_scenes=_migrate_narrative_scenes(data),
            scene_summaries=data.get("scene_summaries", []),
            turn_number=data.get("turn_number", 0),
            consecutive_empty_choice_turns=data.get("consecutive_empty_choice_turns", 0),
            sim_speed=data.get("sim_speed") if data.get("sim_speed") in ("h", "d", "w", "m", "y") else "d",
            pending_speed_change=data.get("pending_speed_change") if data.get("pending_speed_change") in ("h", "d", "w", "m", "y") else None,
            same_scene_streak_since_pace_change=int(data.get("same_scene_streak_since_pace_change", 0) or 0),
        )
        if player.health > player.max_health:
            player.health = player.max_health
        player.permanent_injuries = data.get("permanent_injuries", []) or []
        player.skills = data.get("skills", {}) or {}
        player.inventory = data.get("inventory", []) or []
        player.reputation = data.get("reputation", {}) or {}
        player.status_flags = data.get("status_flags", []) or []
        player.relationships = data.get("relationships", {}) or {}
        player.world_facts = data.get("world_facts", {}) or {}
        player.life_events = data.get("life_events", []) or []
        player.case_file = data.get("case_file", []) or []
        player.expenses = data.get("expenses", []) or []
        player.pregnancies = data.get("pregnancies", []) or []
        player.gear = data.get("gear", []) or []
        player.recovering_injuries = data.get("recovering_injuries", []) or []

        # A save written before the two-tier narrative system existed can
        # migrate in with far more than RECENT_SCENES_FULL_TEXT_COUNT scenes
        # (the old flat buffer held up to 2500 words across many scenes,
        # not just the newest few). Trim it down now, at load time, rather
        # than let an oversized full-text tier linger until enough new
        # turns happen to shrink it naturally - the excess oldest scenes
        # are collapsed into one fallback summary each (no AI call
        # available at load time to write a real one) so they aren't
        # silently dropped from memory entirely.
        if len(player.narrative_scenes) > cls.RECENT_SCENES_FULL_TEXT_COUNT:
            excess = player.narrative_scenes[:-cls.RECENT_SCENES_FULL_TEXT_COUNT]
            player.narrative_scenes = player.narrative_scenes[-cls.RECENT_SCENES_FULL_TEXT_COUNT:]
            for old_scene in excess:
                fallback = " ".join(old_scene.split()[:40])
                if len(old_scene.split()) > 40:
                    fallback += "..."
                player.scene_summaries.append({"summary": fallback, "date": "unknown (migrated)"})
            if len(player.scene_summaries) > cls.SCENE_SUMMARY_MAX_COUNT:
                player.scene_summaries = player.scene_summaries[-cls.SCENE_SUMMARY_MAX_COUNT:]

        return player
