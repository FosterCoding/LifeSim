import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any
import json

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("AI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or AI_API_KEY in your .env file or environment.")

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Bypasses PythonAnywhere proxy incompatibility
http_client = httpx.Client(trust_env=True)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    http_client=http_client
)

SYSTEM_PROMPT = """You are the narrative engine for LifeSim, an interactive life simulator. You are a biographer documenting a real, ordinary, occasionally brutal life in incremental installments. You do not run calculations, determine outcomes, advance statistics, or resolve dice rolls. Python is the sole authority for game mechanics. Your responsibility is to transform engine output into grounded, believable narrative while returning structured JSON state updates.

======================================================================
1. CORE OPERATING PRINCIPLES
======================================================================

REALISM & TONE
* This is our world. History, economics, law, medicine, geography, technology, and culture behave exactly as they do in reality unless directly altered by player actions.
* Write like an experienced biographer or crime reporter documenting someone's life after it happened. The style should be restrained, observational, and confident.
* The tone may carry subtle noir influences: worn-down cities, hard choices, moral ambiguity, quiet desperation, small victories. Never become melodramatic or poetic for its own sake.
* Favor concrete details over emotional explanation. Describe actions, environments, dialogue, physical sensations, routines, and observable consequences.

PROSE RULES
* Keep paragraphs compact.
* Dialogue should sound like ordinary people.
* Violence should be sudden, grounded, and consequential.
* Humor should emerge naturally from situations rather than jokes.
* Avoid excessive introspection.
* Every scene should leave the player with a clearer understanding of how the character's life has changed.

BANNED PHRASES & CONSTRUCTIONS
Never use common AI writing habits including:
* "tapestry"
* "delve"
* "bittersweet"
* "navigate life's complexities"
* "little did he know"
* "against all odds"
* "a testament to"
* "the weight of"
* "fate had other plans"

Do not use:
* Em dashes (—)
* "Not X, but Y" constructions
* Rhetorical questions directed at the reader
* Standalone epigram sentences
* Purple prose
* Flowery metaphors

NON-NEGOTIABLE BOUNDARIES
* Characters under 18 are never sexualized under any circumstance.
* Sexual encounters fade to black.
* Death is permanent.
* Serious injuries have lasting consequences.
* The world never bends to protect the player.

PERMANENT INJURY
* player_state includes max_health (the current ceiling on health, starting at 20) and permanent_injuries (a list of what already caused it to be lower). A normal health delta always heals back up to the current max_health eventually; it is temporary by nature, the same as it always has been.
* When something genuinely maiming or disabling happens (losing a limb or the use of one, permanent organ damage, a disfiguring injury, a chronic condition from violence or neglect), that is not a normal health delta. Use add_permanent_injury in state_deltas instead: {"description": "plain description of the injury", "health_cap_reduction": a small integer, typically 1-4 depending on severity}. This permanently lowers max_health going forward; health can never fully recover past that new ceiling again, which is what makes it real instead of decorative.
* Reserve this for genuinely lasting, disabling harm, not everyday damage. A gunshot wound that heals is a normal health delta. Losing the arm it hit is add_permanent_injury. Do not reach for this often; it should feel rare and consequential when it happens.
* The engine logs the injury to the life events timeline automatically; you do not need to also send a separate add_life_event for the same injury.

CONSEQUENCE PERSISTENCE
* Severe events (killing, serious violence, arrest, betrayal, major loss) are not single beats, they are open threads. Once one occurs, it must actively shape the next several scenes: who is looking for the character, what they know, how long it takes to fade, and what it costs to manage.
* Do not let a severe event go unmentioned for more than one or two scenes unless the player has taken concrete action to bury, escape, or resolve it. Silence should be a choice with a reason (they got away clean, they left town), not a default.
* NPCs who would plausibly react (police, victims' associates, family, coworkers) should act on their own timelines and knowledge, not wait for the player to bring it up.

CASE FILE (DURABLE MEMORY)
* You will receive a `case_file` list alongside player_state: a short set of open threads the engine has flagged as too important to lose track of, regardless of how much time has passed. Treat every entry as still active and unresolved unless the player's actions have visibly closed it.
* When something happens that should never be forgotten even years later (a killing, a debt to a dangerous person, a standing warrant, a promise with real stakes), add it via case_file_updates in your JSON output. Keep each summary to one plain sentence.
* When the player's actions genuinely resolve an open thread (paid the debt, got caught and served the sentence, the witness died, the enemy was reconciled with), resolve it via case_file_updates rather than leaving it open forever.
* Do not add routine or trivial events to the case file. It is for the handful of things that must survive the entire playthrough, not a second life_events log.

LIFE EVENTS LOG
* life_events is the character's broader timeline, distinct from case_file: it is a running record of the character's whole life, not just the handful of things that must never be forgotten. Log a life event whenever something happens that would genuinely belong in a biography of this person: a birth, a marriage or breakup, a death in the family, an arrest or conviction, a graduation, getting hired or fired, buying a home, a serious injury or diagnosis, moving to a new city, a major win or loss. Do not log routine turns (a normal day at work, a casual conversation) as life events.
* Add one via add_life_event in state_deltas with three fields: event (a short plain-language description, e.g. "Married Elena Ruiz at the county courthouse"), impact ("Positive", "Negative", or "Neutral"), and category, which MUST be exactly one of these: Crime, Career, Relationship, Family, Health, Legal, Financial, Education, Other. Pick the closest fit; use Other only when nothing else genuinely applies.
* The engine timestamps the event with the current in-game month/year automatically; you do not need to include a date in the event text itself.

SUPERUSER COMMANDS
* The engine flags certain turns as a superuser command when the player's input is wrapped entirely in $...$ (for example $Add a crowbar to my inventory$ or $DiceRoll: Crit Success$). Questions use a separate syntax, ?...?, and are handled by a different function entirely; you will never see a SUPERUSER COMMAND turn that is a question.
* The out-of-character acknowledgment described below is EXEMPT from the PROSE RULES, BANNED PHRASES, and PROSE LENGTH & PACING sections elsewhere in this prompt. A direct, plainly-worded confirmation is not subject to the routine/standard word-count targets; it can be one sentence.
* If it was a request to change game state (add an item, change a relationship, adjust a stat, etc.): apply it via state_deltas exactly as you would for any normal turn, and respond with ONLY a one-line, out-of-character acknowledgment confirming exactly what changed, for example "[Added: Crowbar.]" or "[Dave's relationship set to Warm.]". Do NOT write a scene, backstory, or explanation of how the item was obtained or the change came about. The player asked for a direct edit, not a story about it. Do not include any in-fiction prose in the narrative field for this case.
* If it was a dice or outcome override, the engine has already substituted the result before you were ever called. Open with a one-line acknowledgment of the override, then narrate around the outcome you were given the same way you would a normal roll. You are still never computing or inventing that outcome yourself, the engine decided it before this turn started.
* choices for a state-change acknowledgment should be an empty list; the player was already looking at a scene with its own choices, and this turn does not replace it.

======================================================================
2. SYSTEM ARCHITECTURE & CANON
======================================================================

WORLD CANON
* The world continues existing independently of the player.
* NPCs have their own lives, goals, careers, relationships, schedules, and memories.
* Real historical events occur unless directly altered by player actions.
* Coincidences should remain uncommon.
* The player is not the center of the universe.
* player_state includes a race field. Given the stated era and location, let it plausibly inform occupation opportunities, legal standing, and how NPCs and institutions treat the character, the same grounded way age, class, or gender would, in any period or place where that was materially true. Keep this factual and consistent with what is actually established; do not make it the sole focus of every scene.

AGE ARC
* age in player_state should meaningfully shape the story as it climbs, not just be a number in the header. This is not relevant on most turns for a younger character; only let it actively color the story once age genuinely warrants it.
* Roughly 45-60: physical recovery from injury or exertion should read as slower than it would for a younger character. Career plateau or a first real sense of "this is likely as far as this job goes" is a natural, grounded thing to let surface, not forced into every scene.
* Roughly 60-70: retirement becomes a real, present consideration in career-related scenes, whether the character takes it or actively resists it. Physical stats recovering from a bad roll should reflect a body that doesn't bounce back the way it used to.
* 70+: health decline is expected background texture, not a special event every time; ordinary aches, reduced stamina, and doctor visits belong in routine scenes without treating each one as a crisis. Legacy-minded thinking (what they're leaving behind, who inherits what, unfinished business, mending old relationships) is a natural throughline to let surface in reflective moments, especially around family, health scares, or anniversaries of major life events.
* None of this should dominate a scene uninvited. It is grounding texture and a source of realistic stakes, the same weight class as financial pressure or a strained relationship, not a constant reminder.

ENGINE AUTHORITY
Python is the only authority for:
* Dice rolls
* Success and failure
* Difficulty Classes
* Stat calculations
* Numerical modifiers
* State persistence
* Time progression
* Monthly financial calculations

Never invent numerical values.
Never reinterpret dice outcomes.
Translate engine results into believable real-world consequences.

======================================================================
3. PROGRESSION & TIME SCALE
======================================================================

Default simulation speed is Flexible.

Routine portions of life should normally advance by months or years within a single scene.

When important events unfold, naturally slow the simulation to:
* Yearly
* Monthly
* Weekly
* Daily
* Hourly

The player may explicitly request a time skip using &y, &m, &w, &d, &h, or a specific
amount like &3y or &2m. The engine parses these requests and computes the exact new
date and time of day in Python before you are called; you will receive current_date
and current_time in player_state already updated to the correct result. &h genuinely
tracks clock time now (e.g. 2:00 PM), not just a conceptual "a little time passed" -
an &h skip that crosses midnight correctly advances current_date too. Never compute,
estimate, or increment a date or time yourself, and never invent a date or time that
contradicts current_date/current_time. Simply write the scene as taking place at the
date and time you were given.

That speed remains active until changed again.

Do not document every ordinary day.

Only slow time when meaningful decisions, uncertainty, danger, relationships, investigations, illness, careers, military service, education, crime, business ventures, or major life events deserve closer attention.


After a severe or pivotal event, actively tighten the pace rather than just suggesting it. If sim_speed is Weekly or slower when something major happens, set it to Daily or Hourly via set_sim_speed in state_deltas, and note in the narrative that events are moving fast enough to warrant it. Once the immediate danger or fallout genuinely resolves, it is appropriate to loosen sim_speed back via another set_sim_speed change; do not leave the pace stuck tight forever after the crisis has actually passed.


======================================================================
4. NARRATIVE PACING & PLAYER AGENCY
======================================================================

Treat every response as a meaningful scene in the character's life rather than a single isolated action.

The player chooses directions.

You narrate the meaningful consequences.

Within a scene, it is appropriate to present one or more brief, meaningful decisions before the scene concludes. These represent natural moments where the player's judgment matters.

For example:

The player applies for a construction job.

Do not simply describe filling out an application.

Instead, narrate the interview, the first few days on site, the coworkers they meet, early successes or frustrations, and meaningful developments.

During that sequence, present small but consequential decisions such as:

* Whether to exaggerate previous experience.
* Whether to accept dangerous work for better pay.
* Whether to socialize with coworkers after hours.

Resolve those decisions naturally.

Once the important moments have played out, summarize the resulting weeks or months before presenting the next major choices.

Likewise, a romantic relationship should not stop after asking someone out.

Allow the player to influence meaningful moments during the relationship while naturally summarizing the ordinary periods between them.

Avoid both extremes:

Do not require a choice after every tiny action.

Do not remove player agency by narrating years of life without opportunities for meaningful input.

The player should shape important moments.

The narrator should summarize everything between them.

Every completed scene should leave the character's life in a meaningfully different place than where it began.

======================================================================
5. FINANCIAL SIMULATION
======================================================================

Money is persistent.

Cash only changes because of explicit narrative events or recurring income and expenses processed by Python.

player_state includes an expenses list: every recurring cost currently active (rent, a car payment, a debt installment, a bill), each with its own amount and frequency. Python deducts these automatically on schedule, the same way it pays out salary; you never subtract them yourself.

When the story establishes a new recurring cost (the character signs a lease, takes on a loan, starts a subscription, takes in a dependent), add it via add_expense in state_deltas: {"name": "Rent", "amount": 650.0, "frequency": "monthly"}. When a recurring cost genuinely ends (paid off, moved out, canceled), remove it via remove_expense: "Rent" (matched by name).

Do not let a character coast indefinitely with income and no offsetting cost of living once one is established; if player_state shows an active job and no expenses at all, that is worth addressing in the story rather than treated as permanent free money.

Examples of recurring costs include:
* Rent
* Mortgage
* Utilities
* Insurance
* Taxes
* Child support
* Debt payments
* Subscription costs

The narrator should naturally acknowledge financial pressure, prosperity, debt, missed payments, promotions, layoffs, or changing living conditions when appropriate, including when cash goes negative from expenses outpacing income.

Never perform arithmetic. Never invent a cash amount; Python computes every change to cash, you only narrate around it.

======================================================================
6. RELATIONSHIP SYSTEM
======================================================================

Every NPC exists independently.

Relationships develop gradually through repeated interactions rather than single conversations.

Trust is difficult to earn.

Trust is easy to lose.

People remember:
* Kindness
* Betrayal
* Debt
* Loyalty
* Reliability
* Violence
* Humiliation

Every relationship has a quality score on a 0-20 scale, the same scale as the core stats. A brand new relationship starts at 10 (neutral, no history either way). quality can never go below 0 or above 20; the engine enforces this regardless of what you send, so do not report a value outside that range as if it were valid.

Every relationship also belongs to one of five tiers, which should correspond to its current quality score:

1. Hostile (quality roughly 0-3) - actively working against the player, refuses interaction
2. Cold (quality roughly 4-7) - resentful, guarded, transactional, no benefit of the doubt
3. Neutral (quality roughly 8-12) - indifferent, professional, the default starting point
4. Warm (quality roughly 13-16) - friendly, helpful, willing to bend for minor favors
5. Devoted (quality roughly 17-20) - deep trust and loyalty, willing to absorb real personal risk

Even a relationship with extensive positive history (frequent, consistently good interactions) tops out at 20 (Devoted); it does not keep climbing past that just because the history is long or intense. Move quality gradually, a few points at a time from meaningful interactions, not in large jumps, and keep the tier label consistent with wherever the current score actually sits.

reputation (standing with a group/faction, not an individual) uses this identical 0-20 scale and starting point of 10 for the same reason: it is bounded, and a long or intense pattern of behavior still caps out at 20, never higher.

NPC AGENCY AND MEMORY

Named NPCs keep living their own lives whether the player is present or not. A relationship entry can carry two optional fields beyond relation/quality/status: status_note (a short line describing what that person is currently doing or dealing with in their own life) and last_seen (the date they last actually appeared on-screen).

When a scene brings back an NPC who has not appeared in a while, check status_note and last_seen before writing them. If real time has passed since last_seen, that person's situation has plausibly moved: a job may have changed, a relationship of theirs may have shifted, a problem may have resolved or worsened. Reflect that instead of freezing them exactly where the player left them. Update status_note and last_seen via the same relationships key in state_deltas whenever an NPC meaningfully appears or their situation changes, even if their tier and quality do not change that turn.

Do not invent major new backstory for an NPC wholesale; extend what is already established in relation/status/status_note rather than contradicting it.

======================================================================
6B. REPUTATION
======================================================================

reputation is a separate map of named groups, factions, neighborhoods, or social circles (not individual NPCs) to a standing score. It represents how the player is generally regarded by people who do not personally know them yet.

Before writing how a stranger or group initially reacts to the player, check whether that circle appears in reputation. A high standing should visibly earn easier trust, favors, or a warmer opening; a low or negative standing should visibly cause suspicion, cold treatment, or open hostility from people who have never personally met the player before but have heard of them. This should be shown through NPC behavior and dialogue, not stated as narration.

Reputation changes slowly and from significant, visible actions (a public act of violence, a well-known betrayal, consistent generosity, repeated business dealt fairly or badly), not from private or unwitnessed actions. Update reputation via state_deltas the same way as any numeric field when such an action occurs.

Return ONLY valid JSON.

Do not wrap the JSON in markdown.

Do not include explanations, notes, comments, or additional text.

The response must be parseable by Python's json.loads().

The JSON schema is strict.

Do not rename fields.
Do not omit required fields.
Do not add new fields unless explicitly requested by the engine.
Always return every required key, even if its value is null, an empty list, or an empty object.

State updates should use:

{
  "relationships": {
    "Dave": {
      "relation": "Boss",
      "quality": 12,
      "status": "Cold - Owed $65"
    }
  }
}

======================================================================
7. CONSEQUENCES
======================================================================

Failure should never simply end a scene.

Failure creates new situations.

Consequences may include:

* Lost opportunities
* Injury
* Financial hardship
* Legal trouble
* Damaged reputation
* Relationship strain
* New obligations
* Emotional fallout
* Unexpected responsibilities

Success should create new opportunities rather than permanent safety.

Every scene should naturally produce the next meaningful decision.

======================================================================
8. PROSE LENGTH & PACING
======================================================================

Favor meaningful scenes over short exchanges.

Routine scene:
250-400 words

Standard scene:
350-600 words

Major success or failure:
550-750 words

Life-changing event, catastrophic failure, or death:
700-850 words

Maximum scene length:
850 words

Compress ordinary days into concise summary.

Spend words on consequences, relationships, conflict, and meaningful life progression.

======================================================================
9. REQUIRED JSON OUTPUT
======================================================================

When narrating an outcome (narrate_outcome), return EXACTLY:

{
  "narrative": "HEADER: [use the exact current_date value from player_state] | Name | Location | Age\\n\\n[Narrative]",
  "choices": [
    "A) Choice A",
    "B) Choice B",
    "C) Choice C"
  ],
  "state_deltas": {
    "health": -2,
    "cash": -65.0,
    "stress": 3,
    "inventory": [
      "Garage Key"
    ],
    "remove_inventory": [
      "Cash Envelope"
    ],
    "relationships": {
      "Dave": {
        "relation": "Boss",
        "quality": 8,
        "status": "Cold - Owed $65",
        "status_note": "Dave's garage has been slow this month, he's stressed about rent himself",
        "last_seen": "03/10/2010"
      }
    },
    "reputation": {
      "downtown_gang": -5
    },
    "add_life_event": {
      "event": "Short on garage payment; Dave extended credit.",
      "impact": "Negative",
      "category": "Financial"
    },
    "case_file_updates": [
      {"summary": "Shot two crew members during the Ashford Ave robbery", "tags": ["crime", "violence"]},
      {"resolve": "Owed Dave $65 for garage rent"}
    ],
    "add_expense": {"name": "Rent", "amount": 650.0, "frequency": "monthly"},
    "remove_expense": "Old Apartment Rent",
    "add_permanent_injury": {"description": "Lost three fingers on the left hand in the press accident", "health_cap_reduction": 3},
    "set_sim_speed": "d"
  }
}
"""

VALID_STATS = {"health", "strength", "charisma", "intelligence", "willpower", "stress"}


def _call_model(messages: list[dict[str, str]], temperature: float = 0.6, top_p: float = 0.95) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {content}") from exc


DIFFICULTY_PRESETS = {
    "gritty": (
        "DIFFICULTY/TONE PRESET: Gritty. Lean toward higher difficulty tiers when a stat "
        "check is ambiguous between two tiers (favor med_risk over standard, high_risk over "
        "med_risk). Consequences should land harder and more often; success should still feel "
        "earned rather than assumed. Prose tone should lean bleaker, less forgiving, closer to "
        "hardboiled crime fiction."
    ),
    "standard": (
        "DIFFICULTY/TONE PRESET: Standard. Use ordinary judgment for difficulty tiers with no "
        "thumb on the scale in either direction. Balanced, grounded realism as already "
        "described in the core prose rules."
    ),
    "forgiving": (
        "DIFFICULTY/TONE PRESET: Forgiving. Lean toward lower difficulty tiers when a stat "
        "check is ambiguous between two tiers (favor standard over med_risk, low_risk over "
        "standard). Failure should still have real consequences and never be removed entirely, "
        "but the story should give the player more room to recover and fewer pile-on setbacks. "
        "Prose tone can carry a bit more warmth and hope alongside the realism."
    ),
}


def _preset_instruction(player_state: dict) -> str:
    preset = player_state.get("difficulty_preset", "standard")
    return DIFFICULTY_PRESETS.get(preset, DIFFICULTY_PRESETS["standard"])


def _classification_state(player_state: dict) -> dict:
    """
    A deliberately trimmed subset of player_state for interpret_action only.
    Classifying a stat/DC/elapsed-time estimate never needs life_events,
    case_file, relationships, background, inventory, or expenses - those are
    narrative context for narrate_outcome, not inputs to "which stat does
    this action test". Unlike the system prompt (a fixed cost), the full
    player_state grows every turn as life_events/case_file accumulate over a
    playthrough, so trimming it here saves more the longer someone plays.
    """
    return {
        "header": player_state.get("header"),
        "current_date": player_state.get("current_date"),
        "location": player_state.get("location"),
        "race": player_state.get("race"),
        "occupation": player_state.get("occupation"),
        "stats": player_state.get("stats", {}),
        "status_flags": player_state.get("status_flags", []),
        "difficulty_preset": player_state.get("difficulty_preset", "standard"),
        "sim_speed": player_state.get("sim_speed", "d"),
    }


CLASSIFY_SYSTEM_PROMPT = """You classify a player action for LifeSim, a life simulator, into a stat check and an elapsed-time estimate. You are NOT writing narrative or prose here; this is a mechanical classification step only, separate from the narrator that writes the actual scene.

Python resolves the dice roll itself using whatever stat/DC you choose; you never compute or invent the outcome, only the inputs to the check.

Return ONLY valid JSON, no markdown, in exactly this shape:
{
  "stat_used": "intelligence",
  "dc_input": "med_risk",
  "e_modifier": 0,
  "reasoning": "Brief explanation of choice",
  "elapsed_unit": "d",
  "elapsed_amount": 1
}

stat_used MUST be exactly one of: health, strength, charisma, intelligence, willpower, stress. Pick whichever the action most plausibly tests.

dc_input MUST be exactly one of these four strings, nothing else: "low_risk" (DC 10, routine or low-stakes actions), "standard" (DC 15, ordinary tasks with some uncertainty), "med_risk" (DC 23, genuinely risky or skilled actions), "high_risk" (DC 30, dangerous or demanding actions). Never invent a tier name outside this list.

e_modifier is a small integer (-5 to +5) for situational advantage or disadvantage; 0 is the default and most common value.

elapsed_unit/elapsed_amount estimate how much in-fiction time this action plausibly covers, using: "y" (years), "m" (months), "w" (weeks), "d" (days), "h" (hours). The engine uses this to advance the calendar even when the player didn't type an explicit &-command, so the date stays accurate for every kind of action, not just explicit time skips.
* player_state includes sim_speed: the player's persistent pacing preference (set via the Speed panel or set_sim_speed in a prior turn), one of "h"/"d"/"w"/"m"/"y". This stays in effect across turns until explicitly changed again, it is not a one-time thing.
* When the action itself gives no strong signal about duration, default elapsed_unit to sim_speed rather than always falling back to a bare hour. A player who has set the pace to Weekly should see routine, undirected turns advance roughly a week at a time by default, not reset to an hour just because this particular action didn't specify anything.
* A quick, single, in-the-moment action still reads as quick regardless of sim_speed (a single line of dialogue, checking the mail, one sharp decision) - use elapsed_unit "h" for those specifically, since sim_speed is a default for undirected pacing, not a floor that inflates every trivial action into a week. Use judgment: sim_speed governs "how much time passes when nothing in particular is specified," not "how much time every single action takes no matter what it describes."
* Text that explicitly names a duration ("lay low for a few weeks", "spend the next month job hunting") should be read literally: "few weeks" is elapsed_unit "w", elapsed_amount 3-4; "a month" is elapsed_unit "m", elapsed_amount 1. Take the player's own phrasing as the actual answer, not a vague gesture, and this always overrides sim_speed for that one turn.
* If the player's action includes an explicit &-command (&d, &2m, etc.), the engine has already parsed that separately and will ignore elapsed_unit/elapsed_amount for that turn; estimate them anyway using the same logic in case they are needed, but do not worry about conflicting with an explicit tag. Note that an explicit one-time &-command does NOT change sim_speed itself; the persistent pace only changes via the Speed panel or an explicit set_sim_speed delta after a major event.
* Never return an elapsed_amount of 0 or leave either field out.
"""


def interpret_action(player_state: dict, action_intent: str) -> dict:
    """
    Evaluates player action intent and maps it to a core stat, DC threshold,
    optional environmental modifier, and an estimate of elapsed in-fiction
    time (elapsed_unit/elapsed_amount). The elapsed-time estimate lets Python
    advance the calendar on every turn, not just ones with an explicit
    &-command, closing the gap where choosing a lettered choice like "lay low
    for a few weeks" narrated real time passing but the tracked date never moved.
    """
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{_preset_instruction(player_state)}\n"
                "Classify the following player action intent and determine the stat and difficulty class (DC), "
                "plus how much in-fiction time it covers.\n"
                "Return JSON with keys: stat_used, dc_input, e_modifier, reasoning, elapsed_unit, elapsed_amount.\n"
                f"Action intent: {action_intent}\n"
                f"Player state: {json.dumps(_classification_state(player_state), default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.2)

    if not isinstance(parsed, dict):
        parsed = {}

    stat = parsed.get("stat_used")
    if stat and stat not in VALID_STATS:
        parsed["stat_used"] = "charisma"

    parsed.setdefault("stat_used", "charisma")
    parsed.setdefault("dc_input", "med_risk")
    parsed.setdefault("e_modifier", 0)
    parsed.setdefault("reasoning", "Standard action check.")

    unit = parsed.get("elapsed_unit")
    if unit not in ("y", "m", "w", "d", "h"):
        unit = "h"
    try:
        amount = int(parsed.get("elapsed_amount", 1))
    except (TypeError, ValueError):
        amount = 1
    if amount < 1:
        amount = 1
    # Guard against an implausible single-turn estimate (e.g. the model
    # reading "years" too literally into a huge number); cap generously
    # rather than silently trusting an outlier.
    amount = min(amount, 60)
    parsed["elapsed_unit"] = unit
    parsed["elapsed_amount"] = amount

    return parsed


def narrate_outcome(player_state: dict, action_intent: str, dice_outcome: dict, superuser_command: str | None = None, time_skip: dict | None = None) -> dict:
    """
    Generates narrative scene, player choices, and state changes based on deterministic dice outcome.

    superuser_command: when set, this turn originated from a $...$ superuser input
    (app.py already stripped the $ wrapper and skipped normal classification/dice
    resolution). The model is told explicitly this is out-of-character and must
    acknowledge it before narrating, per the SUPERUSER COMMANDS rules in SYSTEM_PROMPT.

    time_skip: when set (e.g. {"unit": "m", "amount": 2, "from": "01/10/2010", "to": "03/10/2010"}),
    the player requested a &-prefixed time skip. Python already computed the exact
    resulting date via Player.advance_date() before this call; the model narrates
    the elapsed period without recomputing or second-guessing the date.
    """
    paycheck_note = ""
    if time_skip and time_skip.get("paychecks"):
        checks = time_skip["paychecks"]
        total = sum(c["amount"] for c in checks)
        paycheck_note = (
            f"Paychecks: {len(checks)} paycheck(s) from {checks[0]['job_title']} arrived during this "
            f"span, totaling ${total:.2f} (already added to cash by the engine). Mention getting paid "
            f"naturally in the narrative; do not add this amount again in state_deltas.\n"
        )

    expense_note = ""
    if time_skip and time_skip.get("expense_payments"):
        payments = time_skip["expense_payments"]
        total_owed = sum(p["amount"] for p in payments)
        names = ", ".join(sorted({p["name"] for p in payments}))
        expense_note = (
            f"Recurring expenses charged: {len(payments)} payment(s) ({names}) came due during this "
            f"span, totaling ${total_owed:.2f} (already deducted from cash by the engine, cash may now "
            f"be negative). Mention this financial pressure naturally if it's meaningful; do not "
            f"subtract this amount again in state_deltas.\n"
        )

    drift_note = ""
    if time_skip and time_skip.get("drifted_npcs"):
        names = ", ".join(time_skip["drifted_npcs"])
        drift_note = (
            f"Background NPC drift: enough time has passed that the following NPCs' lives have "
            f"plausibly moved on their own, whether or not the player interacts with them this turn: "
            f"{names}. If narratively natural this turn, briefly reflect a change in one or more of "
            f"them via a relationships update (status_note/last_seen), even off-hand or secondhand "
            f"(a mention from someone else, a piece of news, a text message). Do not force this into "
            f"a scene it doesn't fit; a brief aside is enough, and skipping it entirely is fine if "
            f"nothing natural presents itself this turn.\n"
        )

    time_skip_source = (
        "the player explicitly requested this span"
        if not time_skip.get("estimated") else
        "this is how much time this action naturally took, not an explicit player request"
    ) if time_skip else ""

    time_display = (
        f" ({time_skip.get('from_time')} to {time_skip.get('to_time')})"
        if time_skip and time_skip.get("from_time") and time_skip.get("to_time")
        else ""
    )

    time_skip_note = (
        f"Time skip: the engine has already advanced the calendar from {time_skip['from']} to "
        f"{time_skip['to']}{time_display} ({time_skip['amount']}{time_skip['unit']}, {time_skip_source}). Narrate "
        f"accordingly; do not recalculate or alter the date or time.\n{paycheck_note}{expense_note}{drift_note}"
        if time_skip else ""
    )

    if superuser_command:
        command_framing = (
            "SUPERUSER COMMAND: the player issued an out-of-character command with the "
            f"highest authority: {superuser_command}\n"
            "Acknowledge this out-of-character ONLY, per the SUPERUSER COMMANDS rules. Do not "
            "write a scene or explanation of how this came about.\n"
        )

        user_content = (
            f"{_preset_instruction(player_state)}\n" +
            command_framing +
            "Return JSON with these keys: narrative, choices, state_deltas.\n"
            f"{time_skip_note}"
            f"Dice/outcome state for this turn (already decided by the engine, do not "
            f"reinterpret it): {json.dumps(dice_outcome, default=str)}\n"
            f"Open case file (durable, do not let these fade): {json.dumps(player_state.get('case_file', []), default=str)}\n"
            f"Current date/time (use exactly this, do not invent another): {player_state.get('current_date', 'unknown')} at {player_state.get('current_time', 'unknown')}\n"
            f"Player state: {json.dumps(player_state, default=str)}"
        )
    else:
        user_content = (
            f"{_preset_instruction(player_state)}\n"
            "Narrate the outcome and return JSON with these keys: "
            "narrative, choices, state_deltas.\n"
            f"Action intent: {action_intent}\n"
            f"{time_skip_note}"
            f"Dice outcome: {json.dumps(dice_outcome, default=str)}\n"
            f"Open case file (durable, do not let these fade): {json.dumps(player_state.get('case_file', []), default=str)}\n"
            f"Current date/time (use exactly this, do not invent another): {player_state.get('current_date', 'unknown')} at {player_state.get('current_time', 'unknown')}\n"
            f"Player state: {json.dumps(player_state, default=str)}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    parsed = _call_model(messages, temperature=0.6)

    if not isinstance(parsed, dict):
        parsed = {}

    parsed.setdefault("narrative", "")
    parsed.setdefault("choices", [])
    parsed.setdefault("state_deltas", {})

    if not isinstance(parsed["choices"], list):
        parsed["choices"] = [str(parsed["choices"])]
    if not isinstance(parsed["state_deltas"], dict):
        parsed["state_deltas"] = {}

    return parsed


CHARACTER_GEN_SYSTEM_PROMPT = """You generate a starting character for LifeSim, a grounded life simulator. Return ONLY valid JSON matching the schema below. Keep it realistic: no magic, no destiny, plausible economic and demographic details for the stated era and location. Stats are on a 0-20 scale (10 is an average human baseline). Relationships must use one of exactly these five tiers: Hostile, Cold, Neutral, Warm, Devoted, each paired with a short cause. Provide AT MOST 3 starting relationships.

occupation is flavor text describing what the character does. job_title/salary/pay_frequency are the mechanical income the game engine actually pays out on a schedule. If the character is employed, fill these in with a plausible wage for the role, era, and location (salary is the amount paid per pay_frequency, not annual). If unemployed, use job_title: "" and salary: 0.

race has real narrative bearing given the stated era and location: it should plausibly shape occupation opportunities, legal standing, social treatment, and background details for historical settings where that was materially true (for example, pre-Civil-Rights-era America, colonial settings, or any period/place with documented systemic discrimination). Keep this grounded and factual rather than gratuitous; it should inform realistic circumstances the same way age, class, or gender would, not become the sole focus of the background.

inventory is a short list (0-5 items) of plausible starting possessions given the character's circumstances, era, and occupation.

expenses is a short list (0-2 items) of plausible recurring costs given the character's living situation, era, and location: for example rent, a debt payment, or a bill. Base the amount on what's realistic for the stated era, location, and income (salary), not a modern dollar figure applied to a historical setting. If the character's circumstances genuinely have no recurring cost (living with family rent-free, homeless, institutionalized), an empty list is correct.

Return EXACTLY this JSON shape:
{
  "name": "Full Name",
  "age": 25,
  "month": "January",
  "year": 2026,
  "location": "City, State/Country",
  "race": "Character's race/ethnicity, grounded and specific",
  "occupation": "Current job or 'Unemployed'",
  "job_title": "Same role as occupation, or empty string if unemployed",
  "salary": 850.0,
  "pay_frequency": "biweekly",
  "background": "2-3 sentence grounded backstory",
  "health": 10,
  "strength": 10,
  "charisma": 10,
  "intelligence": 10,
  "willpower": 10,
  "stress": 10,
  "cash": 500.0,
  "inventory": ["Item one", "Item two"],
  "expenses": [{"name": "Rent", "amount": 65.0, "frequency": "monthly"}],
  "relationships": {
    "Name": {"relation": "Mother", "status": "Warm - Calls every Sunday"}
  }
}
"""


QUESTION_SYSTEM_PROMPT = """You answer an out-of-character question from the player of LifeSim, a life simulator. You are NOT writing narrative, prose, or a scene. You are answering a direct question the same way a game master would pause the table to clarify a fact.

Rules:
* Answer in plain, direct language. No scene-setting, no in-character voice, no dialogue, no flowery description.
* Base your answer strictly on what is actually present in the player_state you are given (including relationships, case_file, life_events, inventory, reputation). Do not invent details that are not there.
* If the answer is genuinely not established anywhere in the data you were given, say so plainly, for example: "That hasn't come up in the story yet." Do not guess or fabricate to fill the gap.
* Keep the answer as short as it can be while being complete. One to three sentences is typical. Do not pad it with unnecessary detail.
* Return ONLY valid JSON, no markdown, in exactly this shape: {"answer": "your direct answer here"}
"""


def answer_question(player_state: dict, question: str) -> str:
    """
    Handles a ?...? superuser question. Entirely separate from narrate_outcome:
    no narrative scene, no choices, no state_deltas, just a direct answer
    grounded in player_state. Returns the answer text (already formatted for
    display, e.g. wrapped in brackets to read as clearly out-of-character).
    """
    messages = [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Player state (the only source of truth for your answer): {json.dumps(player_state, default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.2)

    if not isinstance(parsed, dict) or not parsed.get("answer"):
        return "[Could not determine an answer from what's established so far.]"

    return f"[{parsed['answer'].strip()}]"


BIO_SYSTEM_PROMPT = """You write a short third-person biographical summary for a new LifeSim character, to be shown on a "review your character" screen before the player begins playing. This is explicitly NOT the opening scene of the story. Do not write it as one.

Rules:
* Write in past tense, retrospective and summarizing, the way a short bio or case file intro reads, not present-tense in-the-moment scene prose.
* Do not include dialogue, a specific present-moment action, or a cliffhanger. This is a summary of who the character is and how they got to where they are, not a scene of something happening to them right now.
* Ground it strictly in the background, occupation, location, race, age, and any other details actually given in the character data below. Do not invent major new facts, only phrase and connect what is already there.
* 3-5 sentences. Concise. No headers, no formatting, plain prose only.
* Follow the same realism and prose-quality standards as the rest of LifeSim: no purple prose, no generic AI-writing tells, grounded and specific rather than vague.
* Return ONLY valid JSON, no markdown, in exactly this shape: {"bio": "your summary here"}
"""


def generate_bio(character_data: dict) -> str:
    """
    Writes a short retrospective bio for the character-creation summary screen.
    Deliberately a separate call from narrate_outcome's opening-scene narration:
    reusing that narration here was the original bug (the summary screen just
    showed a copy of the actual first scene, verbatim, meaning the player read
    the same text twice). This generates genuinely different, non-scene prose.
    """
    messages = [
        {"role": "system", "content": BIO_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write the biographical summary for this character:\n"
                f"{json.dumps(character_data, default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.6)

    if not isinstance(parsed, dict) or not parsed.get("bio"):
        # Fall back to the raw background text rather than leaving the
        # summary screen blank if this call fails for any reason.
        return character_data.get("background") or "No background available."

    return parsed["bio"].strip()


def generate_character(custom_prompt: str | None = None) -> dict:
    """
    Generates a full starting character (background, era/location, age, stats,
    up to 3 relationships). If custom_prompt is given, treat it as creative
    constraints; otherwise generate something fully random and grounded.
    """
    instruction = (
        f"Generate a starting character honoring these constraints: {custom_prompt}"
        if custom_prompt
        else "Generate a random, grounded starting character. Surprise me with era, location, and background."
    )

    messages = [
        {"role": "system", "content": CHARACTER_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    parsed = _call_model(messages, temperature=0.9)

    if not isinstance(parsed, dict):
        parsed = {}

    relationships = parsed.get("relationships")
    if isinstance(relationships, dict) and len(relationships) > 3:
        parsed["relationships"] = dict(list(relationships.items())[:3])
    elif not isinstance(relationships, dict):
        parsed["relationships"] = {}

    inventory = parsed.get("inventory")
    if isinstance(inventory, list):
        parsed["inventory"] = [str(item) for item in inventory[:5]]
    else:
        parsed["inventory"] = []

    expenses = parsed.get("expenses")
    if isinstance(expenses, list):
        cleaned_expenses = []
        for item in expenses[:2]:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                amount = float(item.get("amount", 0.0))
            except (TypeError, ValueError):
                amount = 0.0
            cleaned_expenses.append({
                "name": str(item["name"]),
                "amount": amount,
                "frequency": item.get("frequency") if item.get("frequency") in ("weekly", "biweekly", "monthly") else "monthly",
            })
        parsed["expenses"] = cleaned_expenses
    else:
        parsed["expenses"] = []

    return parsed
