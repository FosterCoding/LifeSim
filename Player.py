from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple
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

    # Persistent simulation pacing: one of "h", "d", "w", "m", "y". This is the
    # ongoing default the model biases toward every turn until explicitly
    # changed again, distinct from a one-time &-skip which jumps forward once
    # and does not alter this setting.
    sim_speed: str = "d"

    # Persistent simulation state
    skills: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    reputation: Dict[str, int] = field(default_factory=dict)
    debt: float = 0.0
    status_flags: List[str] = field(default_factory=list)

    # Detailed History & Social Tracking (matching visual reference)
    # Structured as: {"Person Name": {"relation": "Mother", "quality": 85, "status": "Living"}}
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Structured as: [{"year": 2024, "age": 18, "event": "Graduated High School", "impact": "Positive"}]
    life_events: List[Dict[str, Any]] = field(default_factory=list)

    # Curated durable memory, survives regardless of how long life_events grows.
    # Structured as: [{"summary": "Shot two crew members during a robbery", "status": "open", "tags": ["crime"]}]
    case_file: List[Dict[str, Any]] = field(default_factory=list)

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
            self.cash += self.salary
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
                self.cash -= amount
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
                })
        open_count = sum(1 for e in self.case_file if e.get("status") == "open")
        if open_count > 20:
            self.case_file = [e for e in self.case_file if e.get("status") == "open"][-20:]
        elif len(self.case_file) > 30:
            resolved = [e for e in self.case_file if e.get("status") == "resolved"]
            open_entries = [e for e in self.case_file if e.get("status") == "open"]
            keep_resolved = max(0, 30 - len(open_entries))
            self.case_file = open_entries + resolved[-keep_resolved:] if keep_resolved else open_entries

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
                    setattr(self, key, new_val)
                # 2. Append to Lists (e.g. "inventory": ["Lockpick"], "status_flags": ["Wounded"])
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

    def export_engine_state(self) -> Dict[str, Any]:
        """Packages the player state into a dictionary for internal tracking or LLM payloads."""
        return {
            "header": f"{self.name} | Age: {self.age} | Date: {self.formatted_date()} | Time: {self.formatted_time()}",
            "current_date": self.formatted_date(),
            "current_time": self.formatted_time(),
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
            "sim_speed": self.sim_speed,
            "skills": self.skills,
            "inventory": self.inventory,
            "relationships": self.relationships,
            "reputation": self.reputation,
            "debt": self.debt,
            "status_flags": self.status_flags,
            "life_events": self.life_events,
            "case_file": [e for e in self.case_file if e.get("status") == "open"],
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
            "sim_speed": self.sim_speed,
            "skills": self.skills,
            "inventory": self.inventory,
            "reputation": self.reputation,
            "debt": self.debt,
            "status_flags": self.status_flags,
            "relationships": self.relationships,
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
            sim_speed=data.get("sim_speed") if data.get("sim_speed") in ("h", "d", "w", "m", "y") else "d",
        )
        if player.health > player.max_health:
            player.health = player.max_health
        player.permanent_injuries = data.get("permanent_injuries", []) or []
        player.skills = data.get("skills", {}) or {}
        player.inventory = data.get("inventory", []) or []
        player.reputation = data.get("reputation", {}) or {}
        player.status_flags = data.get("status_flags", []) or []
        player.relationships = data.get("relationships", {}) or {}
        player.life_events = data.get("life_events", []) or []
        player.case_file = data.get("case_file", []) or []
        player.expenses = data.get("expenses", []) or []
        return player
