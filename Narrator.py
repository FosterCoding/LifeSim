import os
import random
import httpx
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any
import json

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("AI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or AI_API_KEY in your .env file or environment.")


# Planning tier: plan_scene reads case_file/life_events/scene history and
# decides EVERYTHING about a turn before any prose is written - the scene
# outline, era constraints, choices, state_deltas, case_file/life_event
# updates.
#
# DeepSeek V4 Pro (with reasoning) was the original choice here, on the
# theory that this synthesis task would benefit from a model that reasons
# before answering, particularly for catching era anachronisms and
# avoiding banned phrases. However, even at
# "low" reasoning effort (the lowest level tested - "high" was worse and
# separately caused a max_tokens truncation bug), a
# single planning call measured 4,119 OUTPUT tokens at 84.6 tok/s = 48.7s
# of pure generation time, out of which the actual JSON plan is only a
# few hundred tokens. This is the same category of problem that got Kimi dropped from
# the narration tier earlier in this project for an almost identical
# reason (~60s/call, too slow for back-and-forth play) - a reasoning
# model's "think before answering" behavior is fundamentally the wrong
# shape for a turn the player is actively waiting on live, regardless of
# how low the requested effort level is. Reverted to Qwen3-235B, a
# non-reasoning model, with the self-correction behavior reasoning would
# have provided (banned-phrase scrubbing, era-constraint derivation)
# compensated for directly in PLANNING_SYSTEM_PROMPT's wording instead -
# see its ERA ACCURACY section.
PLANNING_MODEL_PRIMARY = os.getenv("PLANNING_MODEL_PRIMARY", "qwen/qwen3-235b-a22b-2507")
PLANNING_MODEL_FALLBACK = os.getenv("PLANNING_MODEL_FALLBACK", "qwen/qwen3.7-flash")

# Writing tier: write_scene takes an already-fully-decided plan (see
# planning tier above) and turns it into prose. This model makes NO
# judgment calls of its own so it doesn't need reasoning-grade capability, only
# reliable instruction-following at speed. Qwen3.7-flash was the original
# primary but it consistently
# hit the max_tokens ceiling (finish_reason "length") on nearly every
# generation regardless of word-count target or how explicitly the prompt
# stated the target as a hard ceiling. Qwen3.7-flash kept
# as the fallback since it is at least a known, working
# option if Ministral's call fails outright.
NARRATION_MODEL_PRIMARY = os.getenv("NARRATION_MODEL_PRIMARY", "mistralai/ministral-14b-2512")
NARRATION_MODEL_FALLBACK = os.getenv("NARRATION_MODEL_FALLBACK", "qwen/qwen3.7-flash")

# Utility tier: classification, extraction, Q&A, and character/bio
# generation - none of these need narration-grade prose, so they route to
# the cheap, fast model first, falling back to Qwen3-235B on failure.
UTILITY_MODEL_PRIMARY = os.getenv("UTILITY_MODEL_PRIMARY", "qwen/qwen3.7-flash")
UTILITY_MODEL_FALLBACK = os.getenv("UTILITY_MODEL_FALLBACK", "qwen/qwen3-235b-a22b-2507")

# Bypasses PythonAnywhere proxy incompatibility
http_client = httpx.Client(trust_env=True)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    http_client=http_client
)

PLANNING_SYSTEM_PROMPT = """You are the planning stage of LifeSim's narrative engine, a life simulator. You are a biographer's outline editor, not the biographer - your job is to decide EVERYTHING about what happens in this turn's scene, in enough concrete detail that a separate writer, working from your plan alone, can turn it into prose without having to make a single judgment call of their own. Python handles all mechanics (dice, stats, money, time) before you are called; you decide the narrative content and every piece of resulting game state.

ERA ACCURACY (check this every scene, not just once): current_date is given in player_state - every object, technology, slang term, cultural reference, and piece of media in the scene must genuinely exist in that year, not just feel vaguely period-appropriate. This is a common failure point - before deciding on anything a character uses to communicate, travel, pay, or research, actively check it against the date. Concretely: no texting, smartphones, social media, streaming, or the modern internet before the mid-2000s (a character in the 1990s uses payphones, landlines, pagers, answering machines, or writes letters - not texts); no widespread internet/email casually referenced before the mid-to-late 1990s, and even then it's slow and unusual, not assumed; only cite songs, movies, shows, technology, or public events that had actually been released or happened by current_date, never something released later even if it feels era-appropriate in spirit; prices, slang, and cultural references should match the actual year, not a vague "past" aesthetic. When in doubt about whether something existed yet, default to the more conservative/older option rather than the modern one. Your era_constraints output (see schema below) is the ONLY thing standing between the writer and an anachronism - list every concrete restriction this scene actually needs, not a generic reminder.

BEFORE YOU ANSWER - CHECK YOUR OWN SCENE_OUTLINE: this is not automatic, you have to do it deliberately as a distinct step after drafting scene_outline and before returning it. Reread what you just wrote and ask: does anything in it require a technology, communication method, or cultural reference that current_date rules out? If yes, rewrite that part with an era-appropriate substitute before finalizing your answer - do not let a first-draft anachronism reach the writer just because it read naturally to write. This self-check is the single most common place this system fails, so treat it as a required step, not an optional pass.

NON-NEGOTIABLE: characters under 18 are never sexualized under any circumstance. Sexual encounters fade to black. Death is permanent.

CASE FILE (DURABLE MEMORY): player_state includes case_file - a short list of open threads too important to lose track of, regardless of how much time has passed (a killing, a debt to a dangerous person, a standing warrant, a promise with real stakes). Treat every entry as still active and unresolved unless the player's actions have visibly closed it - never contradict or silently drop one. When something happens that should never be forgotten even years later, add it via case_file_updates with a one-sentence summary. When the player's actions genuinely resolve one (paid the debt, served the sentence, reconciled), resolve it via case_file_updates. Do not add routine events here - it is for the handful of things that must survive the entire playthrough, not a general log.

LIFE EVENTS LOG: life_events is the character's broader timeline, distinct from case_file - a running record of the whole life, not just the must-never-forget items. Log one via add_life_event whenever something would genuinely belong in a biography of this person: a birth, marriage or breakup, a death in the family, an arrest, a graduation, getting hired or fired, buying a home, a serious injury or diagnosis, moving to a new city, a major win or loss. Use three fields: event (short plain-language description), impact ("Positive", "Negative", or "Neutral"), and category (exactly one of: Crime, Career, Relationship, Family, Health, Legal, Financial, Education, Other). Do not log routine turns.

Both case_file and life_events are established fact the moment they exist - read them before planning a scene and never plan something that contradicts an open case_file entry or a logged life event. If case_file or life_events ever seem to disagree with something in the recent-scenes or earlier-history sections below, case_file and life_events win - they are the durable record; the narrative sections are reference texture, and the earlier-history summaries in particular are a compressed, potentially lossy record of what actually happened, not the authoritative version. The reverse also matters: never plan a scene that references a prior event, promise, conversation, or decision as if it already happened in this playthrough unless it is actually present in case_file, life_events, recent_scenes_full, or earlier_scenes_summary below - do not invent a plausible-sounding piece of backstory or claim something was "already noted" or "already handled" that isn't genuinely there. If it isn't in one of those four places, it didn't happen yet in this character's life.

player_state.inventory is equally established fact, not a suggestion: whatever items are already listed there genuinely belong to the character right now, including on the very first scene of a new playthrough (a player who set up their character's starting possessions manually is relying on those items actually being there). Check inventory before planning any scene that involves what the character is carrying, wearing, or has on hand, and do not write a scene that implies an item isn't there, was never acquired, or contradicts what's listed. You do not need to explicitly narrate every item every turn, but the scene must never act as if inventory is empty or different from what player_state actually says.

Return EXACTLY this JSON shape, nothing else, no markdown, no commentary:

{
  "scene_outline": "A compressed summary of what happens in this scene: who is present, the key beats in order, and how it resolves. This is a list of facts, not a draft - short phrases and plain statements, not full sentences or prose. The writer has NOT seen player_state, case_file, or any of the context you were given, so any specific name, place, or fact the prose needs has to be stated here literally (the writer cannot look anything up) - but state it as a bare fact, not a described scene. This includes inventory: if the scene involves an item the character is carrying, wearing, or using, name that specific item in the outline (checked against player_state.inventory) rather than leaving it implicit - the writer has no way to know what the character has on hand unless you say so here. Aim for the shortest version that still contains everything the writer needs to know, not the richest one.",
  "era_constraints": [
    "Specific, concrete restrictions for this scene given current_date - e.g. 'no cell phones or texting, use the apartment landline', 'pager only, no email'. Empty list if genuinely nothing era-sensitive appears in this scene."
  ],
  "scene_summary": "One or two plain sentences compressing what happens in this scene, written to stand alone as this scene's permanent compressed record once it eventually ages out of the engine's recent-scenes window.",
  "choices": [
    "Short choice A",
    "Short choice B",
    "Short choice C"
  ],
  "state_deltas": {
    "health": -2,
    "cash": -65.0,
    "stress": 3,
    "inventory": ["Garage Key"],
    "remove_inventory": ["Cash Envelope"],
    "relationships": {
      "Dave": {"relation": "Boss", "quality": 8, "status": "Cold - Owed $65"}
    },
    "reputation": {"downtown_gang": -5},
    "add_life_event": {"event": "Short on rent; Dave extended credit.", "impact": "Negative", "category": "Financial"},
    "case_file_updates": [
      {"summary": "Owed Dave $65 for garage rent", "tags": ["financial"]}
    ],
    "add_expense": {"name": "Rent", "amount": 650.0, "frequency": "monthly"},
    "set_sim_speed": "d"
  }
}

add_expense is ONLY for a genuinely new RECURRING financial commitment the character has just taken on going forward - a new rent/lease, a new subscription, a new loan or installment payment, a new bill. frequency must be exactly one of "weekly", "biweekly", "monthly". A single purchase, ticket, meal, or any other one-time transaction is NEVER add_expense - it is a one-time "cash" delta for the amount spent, same as any other spending. Before adding an expense, check player_state's existing expenses list first - if a matching recurring expense already exists (same or clearly equivalent name), do not add it again; only add_expense the first time the commitment is actually taken on. A train ticket, a one-off meal, a single item bought is spending, not a bill - use cash, not add_expense.

scene_outline: this is the single most important field. The writer will follow it literally and will not have access to case_file, life_events, or any other context - if a scene requires knowing a character's name, a past promise, or a specific detail, that detail must be written INTO the outline itself, not merely implied by context only you can see.

choices: 3-4 short, mutually exclusive options when the scene genuinely forks; empty list otherwise. Choices must follow from exactly where scene_outline leaves the character - reread the LAST beat of scene_outline before writing choices, and only offer options that are actually available from that specific moment. A common failure: writing choices anchored to the scene's general premise or an earlier beat (e.g. offering to keep talking to someone who your own outline already has leaving, or referencing an object/conversation the outline moved past) rather than to where the outline actually ends. If the outline's last beat has the character alone, choices must be things that character alone can do next - not things that require someone who just left. relationships/reputation/quality are 0-20 scales. Do not compute dice outcomes or the passage of time yourself - Python already resolved those before you were called. Cash IS yours to decide: any spending, earning, or exchange of money that happens IN the scene (buying something, paying a debt, finding cash, getting paid off the books) must be reflected as a "cash" delta in state_deltas, with a realistic amount for what happened - Python only automatically handles scheduled payroll and recurring expenses, never one-off narrative spending, so if you don't report it, it silently never happens. Any amount you decide - cash deltas, add_expense amounts, prices mentioned in scene_outline - must match current_date's actual era: a 1999 train ticket, a 1970s rent, a 1940s meal should reflect real prices from that period, not present-day figures. Do not default to a modern-feeling round number just because it reads naturally to write.
"""

# Used by write_scene, the writing pass in the plan-then-write split. This
# model makes NO decisions of its own - not about continuity, not about
# era, not about what happens, not about choices or state_deltas (all of
# that was already decided by plan_scene, see PLANNING_SYSTEM_PROMPT).
# Its only job is turning a fully-specified outline into grounded prose.
# Deliberately short: nothing here needs case_file/life_events/choices-
# format instructions, since this model never sees or produces any of
# that.
WRITE_SCENE_SYSTEM_PROMPT = """You are the writer for LifeSim, a life simulator. A planning stage has already decided everything about this turn's scene and given you a scene_outline to follow - your only job is turning that outline into grounded, well-written prose. You are NOT deciding what happens; that decision is already made and binding.

TONE: Write like a restrained biographer or crime reporter - concrete, observational, no melodrama. Never use: "tapestry", "delve", "bittersweet", "navigate life's complexities", "little did he know", "against all odds", "a testament to", "the weight of", "fate had other plans", em dashes, "Not X, but Y" constructions, rhetorical questions to the reader, purple prose. Dialogue should sound like ordinary people. Violence should be sudden, grounded, and consequential. Humor should emerge naturally from situations, not jokes. Avoid excessive introspection.

DIALOGUE BALANCE: dialogue is one tool among several (narration, action, physical detail, interiority), not the default way to convey a scene's content. When scene_outline includes an exchange between characters, do not automatically expand it into extended back-and-forth - render only the lines that actually carry weight (a real decision, a genuine reveal, a turn in the conversation), and compress or summarize the rest through narration instead ("She asked about the money; he said he'd have it by Friday" rather than writing out that entire exchange as quoted lines). A scene that is mostly quoted dialogue front to back is usually a sign whole beats that could be narrated got dramatized instead - especially at shorter target word counts, prioritize what actually needs to be heard in a character's own words over covering a conversation quote by quote.

NON-NEGOTIABLE: characters under 18 are never sexualized under any circumstance. Sexual encounters fade to black. Death is permanent.

FOLLOW THE OUTLINE EXACTLY: the scene_outline you are given in the user message is not a suggestion or a starting point - it is the complete, final content of this scene. Do not add plot beats, characters, objects, or events the outline does not mention. Do not resolve the scene differently than the outline resolves it. Your job is prose craft (pacing, sentence rhythm, sensory detail, dialogue) applied to content that is already fully decided, not independent storytelling.

WORD COUNT IS A HARD CEILING, NOT A SUGGESTION: if the outline contains more beats than comfortably fit within the target word count, compress - summarize or lightly gloss the least important beats in a sentence rather than giving every beat full scene treatment, so the whole scene still lands at the target length. Hitting the word count matters more than depicting every single outline beat at full detail. Do not solve a too-long outline by writing long; solve it by compressing. Never treat "cover everything in the outline" as an excuse to exceed the target.

ERA CONSTRAINTS: you will be given a specific list of era_constraints for this scene (technology, communication methods, cultural references that are or are not allowed). Follow these literally - they were already checked against the actual date by the planning stage specifically so you don't have to independently verify era accuracy yourself.

Your exact word-count target for this scene is given in the user message as "Target word count." This is a hard ceiling - do not exceed it, even if that means compressing outline content more than feels natural. Running short of the target is a minor issue; running over it is not acceptable.

Write ONLY the scene's prose, as plain text - no JSON, no quotes wrapping it, no markdown, no "HEADER:" line or date stamp of any kind (the current date and location are already shown to the player separately, in the game's UI - do not write one yourself, and do not start with a date or scene-heading of your own). Start directly with the narrative prose itself, nothing else before it.
"""


# Injected into the prompt ONLY on a turn app.py has already flagged as a
# superuser command (the player's input was wrapped in $...$) - plan_scene
# appends this to PLANNING_SYSTEM_PROMPT conditionally rather than it living
# there permanently, since it's dead weight on the ~95%+ of turns that
# aren't a superuser command.
SUPERUSER_COMMANDS_BLOCK = """
SUPERUSER COMMANDS
* The engine flags certain turns as a superuser command when the player's input is wrapped entirely in $...$ (for example $Add a crowbar to my inventory$ or $DiceRoll: Crit Success$). Questions use a separate syntax, ?...?, and are handled by a different function entirely; you will never see a SUPERUSER COMMAND turn that is a question.
* The out-of-character acknowledgment described below is EXEMPT from the PROSE RULES, BANNED PHRASES, and PROSE LENGTH & PACING sections elsewhere in this prompt. A direct, plainly-worded confirmation is not subject to the routine/standard word-count targets; it can be one sentence.
* If it was a request to change game state (add an item, change a relationship, adjust a stat, etc.): apply it via state_deltas exactly as you would for any normal turn, and respond with ONLY a one-line, out-of-character acknowledgment confirming exactly what changed, for example "[Added: Crowbar.]" or "[Dave's relationship set to Warm.]". Do NOT write a scene, backstory, or explanation of how the item was obtained or the change came about. The player asked for a direct edit, not a story about it. Do not include any in-fiction prose in the narrative field for this case.
* If it was a dice or outcome override, the engine has already substituted the result before you were ever called. Open with a one-line acknowledgment of the override, then narrate around the outcome you were given the same way you would a normal roll. You are still never computing or inventing that outcome yourself, the engine decided it before this turn started.
* choices for a state-change acknowledgment should be an empty list; the player was already looking at a scene with its own choices, and this turn does not replace it.
"""

VALID_STATS = {"health", "strength", "charisma", "intelligence", "willpower", "stress"}

# Injected only once the character's age actually crosses into the range this
# guidance is about (see AGE_ARC_MIN_AGE below and the age check in
# narrate_outcome) - dead weight on every turn before then, since a 20-year-
# old character has no use for retirement/legacy-thinking guidance. The
# content is unchanged from when it lived inline in SYSTEM_PROMPT; only
# whether it's sent every turn changed.
AGE_ARC_MIN_AGE = 45

AGE_ARC_BLOCK = """
AGE ARC
* age in player_state should meaningfully shape the story as it climbs, not just be a number in the header. This is not relevant on most turns for a younger character; only let it actively color the story once age genuinely warrants it.
* Roughly 45-60: physical recovery from injury or exertion should read as slower than it would for a younger character. Career plateau or a first real sense of "this is likely as far as this job goes" is a natural, grounded thing to let surface, not forced into every scene.
* Roughly 60-70: retirement becomes a real, present consideration in career-related scenes, whether the character takes it or actively resists it. Physical stats recovering from a bad roll should reflect a body that doesn't bounce back the way it used to.
* 70+: health decline is expected background texture, not a special event every time; ordinary aches, reduced stamina, and doctor visits belong in routine scenes without treating each one as a crisis. Legacy-minded thinking (what they're leaving behind, who inherits what, unfinished business, mending old relationships) is a natural throughline to let surface in reflective moments, especially around family, health scares, or anniversaries of major life events.
* None of this should dominate a scene uninvited. It is grounding texture and a source of realistic stakes, the same weight class as financial pressure or a strained relationship, not a constant reminder.
"""

# Appended only when app.py's consecutive_empty_choice_turns counter has
# hit its cap - i.e. the model has already gone too many turns in a row
# without offering a real choice, and Python is no longer treating that as
# a per-turn judgment call it gets to keep making. The actual enforcement
# is app.py refusing to accept another empty result and retrying with this
# block attached - the model still writes the choices, but it does not get
# another turn to opt out of having any.
FORCE_CHOICES_BLOCK = """
MANDATORY CHOICES - THIS TURN
This scene has gone too many turns without offering the player a real decision. This turn is REQUIRED to end with a non-empty "choices" list containing exactly 3 or 4 short, mutually exclusive options with genuinely different consequences. Find or create a real fork in THIS scene: a decision the character is facing right now that the player should make, not a manufactured tangent. Returning an empty choices list this turn is not acceptable.
"""


def _call_model(messages: list[dict[str, str]], temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4000, call_type: str = "utility") -> Dict[str, Any]:
    # max_tokens is set explicitly and generously here. Leaving it unset lets
    # the provider apply its own default completion length, which for a
    # JSON-mode response can be low enough to silently truncate a scene mid-
    # generation - the model has no way to know it's about to be cut off, so
    # this produces exactly the symptom of every scene capping around the
    # same short length no matter what the prose-length instructions ask
    # for, since the cause isn't the prompt at all, it's generation being
    # stopped before the model finishes writing. An 850-word scene plus a
    # choices array plus a state_deltas object, all inside one JSON object,
    # needs real headroom - 4000 tokens covers the longest scenes with
    # margin to spare without being so large it risks runaway output.

    # call_type picks the model tier: "planning" uses Qwen3-235B (see
    # PLANNING_MODEL_PRIMARY's comment for why this is not a reasoning
    # model); "narration" (the writing pass) uses Ministral 14B; "utility"
    # (classification, extraction, Q&A, character/bio gen) uses
    # Qwen3.7-flash. Every tier's fallback only fires on an actual failure
    # (timeout, rate limit, empty/invalid response) - never a quality-
    # based switch.
    if call_type == "planning":
        models_to_try = [PLANNING_MODEL_PRIMARY, PLANNING_MODEL_FALLBACK]
    elif call_type == "narration":
        models_to_try = [NARRATION_MODEL_PRIMARY, NARRATION_MODEL_FALLBACK]
    else:
        models_to_try = [UTILITY_MODEL_PRIMARY, UTILITY_MODEL_FALLBACK]

    last_error: Exception | None = None
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0.1,
                presence_penalty=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"provider": {"sort": "throughput"}},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(f"The model ({model}) returned an empty response.")
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            continue

    # Every model in the tier failed - raise the last error rather than
    # silently returning something callers would mistake for real output.
    raise ValueError(f"All models in tier '{call_type}' failed. Last error: {last_error}") from last_error


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


GENRE_BLOCKS = {
    "realism": (
        "GENRE: Realism. No magic, no supernatural events, no impossible coincidences. "
        "Everything that happens must be plausible in the real world as already described "
        "in the core prose rules."
    ),
    "fantasy": (
        "GENRE: Fantasy. Magic, mythic creatures, and impossible events are a natural, "
        "unremarkable part of this world - treat them with the same grounded, consequence-"
        "driven realism as anything else, not as a gimmick. Do not force fantastical elements "
        "into every turn; an ordinary day is still ordinary."
    ),
    "horror": (
        "GENRE: Horror. Lean into dread, wrongness, and escalating unease where the story "
        "allows it, without abandoning grounded consequence-driven realism elsewhere. Not "
        "every turn needs a scare - restraint and slow-building tension are more effective "
        "than constant horror beats."
    ),
}


def _preset_instruction(player_state: dict) -> str:
    preset = player_state.get("difficulty_preset", "standard")
    genre = player_state.get("genre", "realism")
    preset_text = DIFFICULTY_PRESETS.get(preset, DIFFICULTY_PRESETS["standard"])
    genre_text = GENRE_BLOCKS.get(genre, GENRE_BLOCKS["realism"])
    return f"{preset_text}\n{genre_text}"


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
        "genre": player_state.get("genre", "realism"),
        "sim_speed": player_state.get("sim_speed", "d"),
        # Not used for stat/DC choice, only for the elapsed-time estimate:
        # a short tail of the previous scene lets the classifier recognize
        # when the new action is still inside that same moment (a reaction,
        # a follow-up choice, a continuation of the same exchange) so it
        # doesn't estimate a time skip that would contradict the scene the
        # narrator is about to continue.
        "previous_scene_tail": (player_state.get("recent_scenes_full") or [""])[-1][-400:],
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
  "elapsed_amount": 1,
  "same_scene_continuation": false
}

stat_used MUST be exactly one of: health, strength, charisma, intelligence, willpower, stress. Pick whichever the action most plausibly tests.

dc_input MUST be exactly one of these five strings, nothing else: "auto" (no roll at all, see below), "low_risk" (DC 10, routine or low-stakes actions), "standard" (DC 15, ordinary tasks with some uncertainty), "med_risk" (DC 23, genuinely risky or skilled actions), "high_risk" (DC 30, dangerous or demanding actions). Never invent a tier name outside this list.

Use "auto" whenever the action was never genuinely in doubt: the player is simply stating something they do, not attempting something with a real chance of not working out. Arriving somewhere on time, greeting someone, initiating a conversation, sitting down, handing over an object, doing an established part of a routine, being polite, following through on something already agreed to - these are "auto," not "low_risk." Note this is purely about whether a dice roll happens, not about how much the resulting scene should say - a conversation classified "auto" can still run long and carry real narrative weight if the content itself deserves it (see PROSE LENGTH & PACING); "auto" only means nothing was mechanically in doubt, never "keep this brief." The test is "could this plausibly just fail or go sideways on its own," not "is this a big or small moment." A roll should mean there is actually something at stake in whether it succeeds. Reserve "low_risk" for actions that are easy but still carry some genuine chance of an unwanted wrinkle (a stranger might say no, a lock might stick, a shortcut might be blocked). Most ordinary turns of a life, especially routine social and workplace moments, should be "auto." Do not reach for "auto" to dodge a check that's actually in question just because the activity is common or everyday - a routine task performed under real scrutiny or pressure, or a request that could plausibly be refused, still warrants a real tier.

Judge dc_input strictly on what the player's own action literally does, not on how tense, dangerous, or high-stakes the surrounding scene feels. A scene can be full of danger cues (weapons mentioned, a criminal setting, a menacing NPC, an ominous location) while the player's specific action this turn is still just walking somewhere, listening, or showing up - that stays "auto." The tension of the surroundings is not evidence the action itself is contested; only judge whether this literal action could plausibly fail or go wrong. "Go meet someone to hear what they want" is auto regardless of who they are or where the meeting is - showing up and listening carries no risk of failure by itself. It only becomes a real tier once the player is the one doing something that could go wrong (making an ask, lying, fighting, sneaking, persuading, resisting). previous_scene_tail (if present below) exists only to help judge elapsed time for pacing continuity - never let its mood or content push dc_input to a higher tier than the literal action alone would warrant.

Worked contrasts:
* "Arrive at work on time and greet the boss" -> auto. Nothing about showing up and saying good morning can plausibly fail on its own.
* "Arrive at work and ask the boss for a raise" -> a real tier (med_risk or higher). The boss could say no; there is something genuinely at stake.
* "Say good morning to a coworker" -> auto. "Strike up a conversation to find out if a coworker is hiring for their side business" -> a real tier. Small talk with no ask behind it is auto; small talk in service of extracting information, a favor, or a commitment from someone is not, even if it looks like ordinary chat on the surface.
* "Hand the cashier exact change" -> auto. "Talk the cashier into a discount they're not supposed to give" -> a real tier.
* "Walk home the usual way" -> auto. "Take a shortcut through an alley you don't know" -> low_risk or higher.
* "Follow the recipe you've made a hundred times" -> auto. "Try a recipe for the first time for guests" -> low_risk.
* "Head to meet a contact to hear what they want, in a tense or dangerous setting" -> still auto. Walking to a meeting and listening to a proposition is not itself contested, no matter how ominous the neighborhood or how illegal the subject matter turns out to be. "Negotiate the terms of the deal Joseph is proposing" or "decide whether to take the deal and say so" -> a real tier, once the player is actually the one acting on something that could go wrong.
If the scene as written gives the player nothing to actually win or lose (no one could plausibly refuse, resist, notice, or object), it is auto even if it involves a boss, a stranger, money, a job, or a dangerous setting, none of which by themselves push an action off auto.

e_modifier is a small integer (-5 to +5) for situational advantage or disadvantage; 0 is the default and most common value.

elapsed_unit/elapsed_amount estimate how much in-fiction time this action plausibly covers, using: "y" (years), "m" (months), "w" (weeks), "d" (days), "h" (hours). The engine uses this to advance the calendar even when the player didn't type an explicit &-command, so the date stays accurate for every kind of action, not just explicit time skips.
* player_state includes sim_speed: the player's persistent pacing preference (set via the Speed panel or set_sim_speed in a prior turn), one of "h"/"d"/"w"/"m"/"y". This stays in effect across turns until explicitly changed again, it is not a one-time thing.
* When the action itself gives no strong signal about duration, default elapsed_unit to sim_speed rather than always falling back to a bare hour. A player who has set the pace to Weekly should see routine, undirected turns advance roughly a week at a time by default, not reset to an hour just because this particular action didn't specify anything.
* A quick, single, in-the-moment action still takes little in-fiction TIME regardless of sim_speed (checking the mail, one sharp decision, a brief exchange) - use elapsed_unit "h" for those specifically, since sim_speed is a default for undirected pacing, not a floor that inflates every trivial action into a week. This governs elapsed time only, never prose length: a single exchange of dialogue can still be a full scene worth real words if it's where the actual weight of the turn lives (a confrontation, a confession, a negotiation) - a conversation taking five minutes of story-time and a conversation taking five hundred words of prose are unrelated facts. Use judgment: sim_speed governs "how much time passes when nothing in particular is specified," not "how much time every single action takes no matter what it describes," and neither one governs how much the scene should actually say.
* Text that explicitly names a duration ("lay low for a few weeks", "spend the next month job hunting") should be read literally: "few weeks" is elapsed_unit "w", elapsed_amount 3-4; "a month" is elapsed_unit "m", elapsed_amount 1. Take the player's own phrasing as the actual answer, not a vague gesture, and this always overrides sim_speed for that one turn.
* If the player's action includes an explicit &-command (&d, &2m, etc.), the engine has already parsed that separately and will ignore elapsed_unit/elapsed_amount for that turn; estimate them anyway using the same logic in case they are needed, but do not worry about conflicting with an explicit tag. Note that an explicit one-time &-command does NOT change sim_speed itself; the persistent pace only changes via the Speed panel or an explicit set_sim_speed delta after a major event.
* Never return an elapsed_amount of 0 or leave either field out.
* player_state includes previous_scene_tail: the end of the last scene's prose, for this purpose only (it is not narrative context to draw on otherwise). If the current action is plainly a direct continuation of that same moment (a reaction to something just described, a choice among options just presented, a follow-up in the same exchange, or another beat of a single continuous activity like a journey, conversation, or task already in progress) - set same_scene_continuation to true, and use elapsed_unit "h" with a small amount. Only estimate longer spans, and set same_scene_continuation to false, when the action's own content genuinely implies a break from that moment (explicit duration, or a clear jump like leaving the scene to go do something else). same_scene_continuation is the authoritative signal for this - Python will use it (not just the size of elapsed_amount) to decide whether the player's persistent pace setting is allowed to force a longer jump than you estimated, so it must accurately reflect whether this turn is still inside the same continuous beat, not just be a formality alongside a small elapsed_amount.
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
                "Return JSON with keys: stat_used, dc_input, e_modifier, reasoning, elapsed_unit, elapsed_amount, same_scene_continuation.\n"
                f"Action intent: {action_intent}\n"
                f"Player state: {json.dumps(_classification_state(player_state), default=str)}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.2, call_type="utility")

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

    # Coerce to a real bool with a safe default of False (i.e. "not a
    # continuation, sim_speed's floor is allowed to apply") - a model that
    # omits the field entirely, or a provider that returns "true"/"false"
    # as a string instead of a JSON boolean, should never silently be
    # treated as bypassing the floor by accident.
    parsed["same_scene_continuation"] = parsed.get("same_scene_continuation") is True

    return parsed


EXTRACTION_SYSTEM_PROMPT = """You are a mechanical extraction pass for LifeSim, a life simulator. You are NOT writing narrative, prose, dialogue, or a scene. A separate narrator already wrote the scene text you are given below; your only job is to read it and report which of a fixed set of structured game-state events it describes, if any.

You are the ONLY place these events get decided - the narrator that wrote the scene does not and cannot trigger any of them itself. If a scene clearly establishes one of these events and you fail to report it, that event simply never happens in the game, permanently. Read carefully and err toward reporting a real, clearly-described event rather than omitting it out of caution - but never invent or infer an event the scene does not actually support. Silence (returning nothing for a category) is correct and expected on most turns; most scenes describe no new mechanical event at all.

Base your answer strictly on the scene text provided. Do not use outside knowledge of the characters or story beyond what this scene text and the player_state summary say. Do not re-report something player_state shows is already true (e.g. a crew member who is already listed as active) - only report NEW events this scene introduces.

Return ONLY valid JSON, no markdown, matching exactly this shape (omit any key entirely if that category has nothing to report this turn - do not include empty objects/lists/nulls for categories with nothing to report):

{
  "new_pregnancy": {"partner": "Elena Ruiz"},
  "new_expense": {"name": "Rent", "amount": 650.0, "frequency": "monthly"},
  "removed_expense": "Rent",
  "new_gear": {"name": "Lockpick set", "e_modifier": 2, "applies_to": "intelligence"},
  "removed_gear": "Lockpick set",
  "new_recovering_injury": {"description": "Sprained wrist", "stat": "strength", "penalty": 2, "recovery_days": 21},
  "new_permanent_injury": {"description": "Lost two fingers in a press accident", "health_cap_reduction": 3},
  "new_relationships": {"Ray": {"relation": "Corner Boy", "quality": 10}},
  "new_world_facts": {"Rosa's Diner": "The family restaurant on Elm Street, run by her father until it closed in 1998."}
}

RULES FOR EACH CATEGORY:

new_pregnancy: report only when the scene explicitly establishes conception this turn. Check player_state's pregnancies first - do not report a new one if an unresolved pregnancy already exists for the same partner.

new_expense / removed_expense: a new or ended recurring real-world cost (rent, a loan, a subscription, a dependent).

new_gear / removed_gear: equipment with an obvious mechanical use in a future check (weapon, lockpicks, tools, disguise). applies_to must be one of health, strength, charisma, intelligence, willpower. e_modifier 1-3 for most items, higher only for something narratively exceptional.

new_recovering_injury: a temporary impairment with a recovery timeline. new_permanent_injury: lasting harm that lowers max_health going forward - use sparingly, only for genuinely permanent damage the scene clearly establishes as such.

new_relationships: report EVERY named person this scene introduces who should be tracked going forward - anyone given a name who has direct dialogue/interaction with the player, or who is established as a recurring counterpart in an ongoing arrangement, conflict, deal, rivalry, or relationship of any kind (an associate, a rival, a contact, a partner, a supplier, an enemy, a love interest, a boss, a coworker who reappears). Do not re-report someone player_state's relationships already lists. A person named once with no real interaction and no expectation of reappearing does not need an entry. quality is 0-20 (10 = neutral); set it to reflect the scene's own established dynamic on first contact, defaulting to 10 only if the scene gives no clear signal either way.

new_world_facts: report a standing detail about the world - a specific place, an object, a piece of history - that the scene establishes as something worth remembering exactly, not just atmosphere. The test is: would getting this detail wrong in a future scene be a real, noticeable continuity error (a place's name, an address, what happened to a childhood home, the origin of a scar or an heirloom, the name a business used to go by) - if so, report it as a short key (a name/short label) mapped to one plain sentence stating the fact. Do not report vague mood or one-off scenery with no lasting specificity. Do not re-report a fact player_state's world_facts already lists under the same or a clearly matching key.

Every dollar figure, percentage, and numeric estimate you provide must be a real, considered value appropriate to the scene's content, era, and location - never a placeholder or default copied mechanically from the examples above.
"""


def extract_mechanical_deltas(player_state: dict, scene_text: str) -> dict:
    """
    The dedicated, narrow extraction pass that is the ONLY place mechanical
    trigger events (a pregnancy, a new/ended expense, gear acquired, an
    injury, a new relationship) get decided. Previously these lived as
    "remember to do this" instructions buried inside the same single call
    that also had to write good prose, manage tone, follow banned-phrase
    rules, track relationships, and everything else at once - which meant a
    real, clearly-described event could and did get silently skipped simply
    because the model's attention was elsewhere that turn. Splitting this
    into its own small, focused call whose only job is "read this scene,
    report structured events" makes the trigger decision itself far more
    reliable, since there's nothing else competing for the model's
    attention in this call.

    Called by app.py after every narrate_outcome call that produced real
    scene prose (never for out-of-character question/command turns, which
    have no scene to extract from). Returns a dict of only the categories
    that fired this turn; Player.apply_extracted_deltas (Player.py) is
    responsible for turning this into the actual mechanical calls, with the
    same validation/clamping every other delta path already goes through -
    this function only ever decides WHAT happened, never how the engine
    applies it.
    """
    if not scene_text or not scene_text.strip():
        return {}

    trimmed_state = {
        "location": player_state.get("location"),
        "current_date": player_state.get("current_date"),
        "cash": player_state.get("cash"),
        "relationships": player_state.get("relationships", {}),
        "world_facts": player_state.get("world_facts", {}),
        "pregnancies": player_state.get("pregnancies", []),
        "gear": player_state.get("gear", []),
        "expenses": player_state.get("expenses", []),
    }

    user_content = (
        f"Scene text to extract from:\n{scene_text}\n\n"
        f"Relevant current player_state (for checking what already exists, so you don't "
        f"re-report it): {json.dumps(trimmed_state, default=str)}"
    )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        # 3500, raised again from 2500 (originally 1500) - real production
        # logs showed this call hitting finish_reason "length" even before
        # the first bump. Temperature also dropped further (0.2 -> 0.1) as
        # a second lever being tested alongside the token increase: lower
        # temperature won't directly control output LENGTH (that's a
        # sampling-randomness knob, not a verbosity one), but it's cheap to
        # test together with more headroom before concluding the model
        # itself needs to change. If length still gets hit at 3500 with a
        # near-deterministic temperature, that's real evidence the
        # verbosity is a formatting habit this model has, not something
        # temperature can fix, and the model itself is the next lever.
        parsed = _call_model(messages, temperature=0.1, max_tokens=3500, call_type="utility")
    except Exception:
        # Extraction failing should never break the turn - the scene the
        # player already read is still valid, it just means any mechanical
        # event in it goes uncaught this turn. Fail closed to "nothing
        # happened" rather than raising and losing the whole turn's result.
        return {}

    return parsed if isinstance(parsed, dict) else {}


# Maps a rolled DC back to the tier name it came from. Dice.py is the sole
# owner of these numbers (see difficulty_classes there); duplicated here
# only for the reverse lookup, since dice_outcome only ever carries the
# resolved DC, not the tier name it started as.
_DC_TO_TIER = {10: "low_risk", 15: "standard", 23: "med_risk", 30: "high_risk"}

# midpoint-ish target within each tier's old range, used as a single exact
# number rather than leaving the model to pick anywhere in a range.
_TIER_WORD_TARGETS = {
    "auto": 225,
    "low_risk": 225,
    "standard": 300,
    "med_risk": 300,
    "high_risk": 400,
    "life_changing": 400,
}


# A time-skip's own word-count FLOOR, independent of dice tier. Dice tier
# answers "how dramatic was the immediate outcome"; elapsed span answers
# "how much life happened in between" - these are different axes, and a
# routine/auto outcome spanning a whole month still has a month of life to
# actually narrate, not the same ~350 words as an auto outcome spanning an
# hour. Applied as a floor (max against the dice-tier target), not a
# replacement, so a dramatic outcome during a long skip still gets at least
# its own higher dice-tier target, never LESS room than the span alone
# would warrant.
_TIME_SPAN_WORD_FLOORS = {
    "h": 0,      # no floor - same-hour turns are governed by dice tier alone
    "d": 0,      # a single day, same reasoning
    "w": 250,    # scaled proportionally with the tier-target reduction (0.444x of the prior 550) - a week of elapsed life still needs more room than a routine same-day turn, just not as much as before
    "m": 335,    # scaled proportionally (0.444x of the prior 750)
    "y": 420,    # scaled proportionally (0.444x of the prior 950)
}

# Choices should get COARSER as elapsed time gets LARGER - a turn spanning
# a year should never fork on a single line of dialogue the way an hour-
# scale turn can, and a turn spanning an hour shouldn't offer a choice as
# broad as "focus on your career this year." Computed in Python from the
# same time_skip.unit that already drives _target_word_count and
# era_constraints, rather than left as a vague "use good judgment" note -
# the same reasoning as every other deterministic-where-possible value in
# this file: a rule stated as a concrete instruction with a real example
# holds up under real generation far better than an implied one.
_CHOICE_GRANULARITY_BY_SPAN = {
    "h": "Specific and immediate - a concrete action or line available right now in this exact moment (what to say, what to do next, in this scene).",
    "d": "Specific and immediate - a concrete action or line available right now in this exact moment (what to say, what to do next, in this scene).",
    "w": "Broader than a single line or action - what the character generally does with or how they approach the week ahead (e.g. \"Spend the week job-hunting\" / \"Spend it patching things up with Dave\"), not a specific moment-to-moment choice.",
    "m": "A general direction or priority for the month, not a specific scene or action (e.g. \"Focus on saving money\" / \"Chase the promotion\" / \"Deal with the Marcus situation once and for all\").",
    "y": "A real life-direction choice at the scale a year actually operates on (e.g. \"Leave the city\" / \"Commit to the relationship\" / \"Go all-in on the business\"), not anything that reads like a single scene's decision.",
}

_SPEED_LABELS = {
    "h": "hour-by-hour",
    "d": "day-by-day",
    "w": "week-by-week",
    "m": "month-by-month",
    "y": "year-by-year",
}


def _pace_change_context_note(new_speed: str) -> str:
    """
    Built from Player.pending_speed_change. Earlier versions of this were
    instructions asking the model to conclude/switch scenes on a specific
    schedule - real production evidence (direct prompt/response log
    inspection) confirmed those were delivered correctly and simply not
    complied with. The actual pace enforcement now happens in Python (see
    app.py's _apply_sim_speed_floor/force_ignore_same_scene), which can
    force a real time-skip regardless of what the model wants to narrate.
    This is no longer an instruction trying to get compliance - it's
    informational context, told to the model so that WHEN Python has
    forced a jump forward, the model has an honest explanation for why
    time suddenly advanced, rather than narrating around an unexplained
    gap. It does not need to persuade the model of anything anymore.
    """
    label = _SPEED_LABELS.get(new_speed, new_speed)
    return (
        f"\nNote: the player recently changed their ongoing pace to {label}. If this turn's dice/"
        f"time-skip info below shows a larger jump than the immediately preceding turns, that is why - "
        f"narrate the transition naturally (the previous scene concluding, time passing) rather than "
        f"treating the jump as unexplained.\n"
    )


def _choice_granularity_instruction(time_skip: dict | None) -> str:
    """
    Returns the granularity guidance line for this turn's choices, keyed
    off time_skip.unit the same way _target_word_count and era_constraints
    already are. No time_skip (or an hour/same-day one) uses the finest,
    most specific grain; longer spans step up through week/month/year.
    """
    unit = (time_skip or {}).get("unit", "h")
    return _CHOICE_GRANULARITY_BY_SPAN.get(unit, _CHOICE_GRANULARITY_BY_SPAN["h"])


def _target_word_count(dice_outcome: dict, time_skip: dict | None = None) -> int:
    """
    Computes an EXACT target word count for this turn's scene, in Python,
    from dice_outcome alone - dice_outcome is already fully decided by the
    time narrate_outcome is called, so nothing here is a guess. This
    replaces the old system of giving the model a 100-450 word RANGE per
    tier and trusting it to land somewhere reasonable within it, which in
    practice meant it consistently landed at the low end (170-200 words)
    regardless of stakes - a range is still a choice, and the model kept
    making the same lazy choice. A single number removes that choice
    entirely, the same fix already applied to sim_speed and the mechanical
    extraction triggers: anything that can be computed deterministically
    should be, rather than left as prompt-only guidance the model can
    quietly under-deliver on turn after turn.

    time_skip, when provided, applies _TIME_SPAN_WORD_FLOORS as a floor on
    top of the dice-tier target - see that dict's comment for why elapsed
    span and dice tier are tracked as separate axes rather than one
    replacing the other.
    """
    if not isinstance(dice_outcome, dict):
        base = _TIER_WORD_TARGETS["standard"]
    elif dice_outcome.get("auto"):
        base = _TIER_WORD_TARGETS["auto"]
    else:
        outcome = dice_outcome.get("outcome", "")
        if outcome in ("great_success", "Crit fail"):
            base = _TIER_WORD_TARGETS["life_changing"]
        else:
            dc = dice_outcome.get("dc")
            tier = _DC_TO_TIER.get(dc, "standard")
            base = _TIER_WORD_TARGETS[tier]

    if time_skip and isinstance(time_skip, dict):
        span_unit = time_skip.get("unit", "h")
        floor = _TIME_SPAN_WORD_FLOORS.get(span_unit, 0)
        base = max(base, floor)

    return base


def _build_planning_context(player_state: dict, action_intent: str, dice_outcome: dict, superuser_command: str | None = None, time_skip: dict | None = None) -> dict:
    """
    Assembles everything plan_scene needs to make its call: all the time-
    skip notes, continuity framing, and the final user_content string.
    This is the single call that decides everything about a turn (scene
    outline, era constraints, choices, state_deltas) - write_scene, the
    separate writing pass, never sees any of this context directly, only
    the finished plan's scene_outline/era_constraints/target word count.

    Returns a dict with: user_content (str), effective_system_prompt (str,
    PLANNING_SYSTEM_PROMPT + conditional blocks), target_words (int), and
    a few raw fields (dice_outcome, player_state) passed through for
    convenience so callers don't need to re-derive them.
    """
    paycheck_note = ""
    if time_skip and time_skip.get("paychecks"):
        checks = time_skip["paychecks"]
        total = sum(c["amount"] for c in checks)
        paycheck_note = (
            f"Paychecks: {len(checks)} paycheck(s) from {checks[0]['job_title']} arrived during this "
            f"span, totaling ${total:.2f} (already added to cash by the engine). Include getting paid "
            f"naturally in the scene_outline; do not add this amount again in state_deltas.\n"
        )

    expense_note = ""
    if time_skip and time_skip.get("expense_payments"):
        payments = time_skip["expense_payments"]
        total_owed = sum(p["amount"] for p in payments)
        names = ", ".join(sorted({p["name"] for p in payments}))
        expense_note = (
            f"Recurring expenses charged: {len(payments)} payment(s) ({names}) came due during this "
            f"span, totaling ${total_owed:.2f} (already deducted from cash by the engine, cash may now "
            f"be negative). Include this financial pressure in the scene_outline if it's meaningful; do "
            f"not subtract this amount again in state_deltas.\n"
        )

    drift_note = ""
    if time_skip and time_skip.get("drifted_npcs"):
        names = ", ".join(time_skip["drifted_npcs"])
        drift_note = (
            f"Background NPC drift: enough time has passed that the following NPCs' lives have "
            f"plausibly moved on their own, whether or not the player interacts with them this turn: "
            f"{names}. If it fits, plan a brief reflection of a change in one or more of them via a "
            f"relationships update (status_note/last_seen), even off-hand or secondhand (a mention from "
            f"someone else, a piece of news, a text message). Do not force this into a scene it doesn't "
            f"fit; skipping it entirely is fine if nothing natural presents itself this turn.\n"
        )

    births_note = ""
    if time_skip and time_skip.get("births"):
        birth_lines = []
        for b in time_skip["births"]:
            partner_txt = f" with {b['partner']}" if b.get("partner") and b["partner"] != "Unknown" else ""
            birth_lines.append(f"A birth occurred{partner_txt} on {b['date']}.")
        births_note = (
            "LIFE EVENT (already decided by the engine): " + " ".join(birth_lines) + " Plan the scene_outline "
            "to depict this as genuinely happening this turn, and give the child a name if one isn't already "
            "established. A life_events entry and a relationships entry for the child already exist; do not "
            "duplicate them.\n"
        )

    stale_threads_note = ""
    if time_skip and time_skip.get("stale_threads"):
        thread_lines = [t["summary"] for t in time_skip["stale_threads"]]
        stale_threads_note = (
            "OPEN THREAD DUE (already selected by the engine, not optional and not a suggestion to "
            "weigh - this must actually surface in this scene, in some concrete form): " +
            " | ".join(thread_lines) + ". This has gone untouched for a long time and the engine has "
            "flagged it as due. Plan a real, visible consequence or development of at least one of "
            "these threads into THIS scene's outline - a consequence catching up with the character, "
            "someone calling it in, a reminder that forces a decision, a complication arising from it "
            "having been ignored. A passing mention is not enough; something has to actually happen "
            "because of it. If the thread genuinely resolves as a result, close it via case_file_updates "
            "the same way you would any other resolution.\n"
        )

    recovery_note = ""
    if time_skip and time_skip.get("recovered_injuries"):
        descs = ", ".join(i["description"] for i in time_skip["recovered_injuries"])
        recovery_note = (
            f"Recovery: the following temporary injuries have healed and no longer apply a penalty: "
            f"{descs}. Reflect this in the scene_outline if relevant (relief, resumed activity); do not "
            f"keep treating them as ongoing.\n"
        )

    time_skip_source = (
        "the player explicitly requested this span"
        if not time_skip.get("estimated") else
        "this is how much time this action naturally took, not an explicit player request"
    ) if time_skip else ""

    if time_skip and time_skip.get("same_day"):
        calendar_line = (
            f"No real time skip occurred this turn - the action stayed within the same day. "
            f"However, a background system resolved something during this turn regardless (see below); "
            f"plan it as happening now, in the current scene, not as a jump forward in time.\n"
        )
    elif time_skip:
        calendar_line = (
            f"Time skip: the engine has already advanced the calendar from {time_skip['from']} to "
            f"{time_skip['to']} ({time_skip['amount']}{time_skip['unit']}, {time_skip_source}). Plan "
            f"accordingly; do not recalculate or alter the date.\n"
        )
    else:
        calendar_line = ""

    time_skip_note = (
        f"{calendar_line}"
        f"{paycheck_note}{expense_note}{drift_note}{births_note}{stale_threads_note}{recovery_note}"
        if time_skip else ""
    )

    override_skipped_note = (
        f"Note: the player had a dice-outcome override active for this turn, but this action never "
        f"rolled a check in the first place (it was routine/certain), so the override had nothing to "
        f"apply to and was skipped. Plan this turn normally as a certain, uncontested action - do "
        f"not manufacture tension, risk, or a twist just because an override was requested; do not "
        f"mention the override itself in the scene_outline, it is out-of-character bookkeeping only.\n"
        if isinstance(dice_outcome, dict) and dice_outcome.get("override_skipped")
        else ""
    )

    target_words = _target_word_count(dice_outcome, time_skip)
    choice_granularity = _choice_granularity_instruction(time_skip)

    if superuser_command:
        command_framing = (
            "SUPERUSER COMMAND: the player issued an out-of-character command with the "
            f"highest authority: {superuser_command}\n"
            "Plan the acknowledgment described in the SUPERUSER COMMANDS rules. The scene_outline for "
            "this turn should be that one-line acknowledgment itself (or, for a dice-override, a short "
            "outline of the scene around the given outcome) - not a full scene.\n"
        )

        user_content = (
            f"{_preset_instruction(player_state)}\n" +
            command_framing +
            "Return JSON matching the schema in your instructions.\n"
            f"Target word count if this turn narrates a real scene: {target_words} words. (Not applicable "
            f"for a plain state-change acknowledgment, which stays one line per the SUPERUSER COMMANDS "
            f"rules.)\n"
            f"Choice granularity for this turn, if choices is non-empty: {choice_granularity}\n"
            f"{time_skip_note}"
            f"Dice/outcome state for this turn (already decided by the engine, do not "
            f"reinterpret it): {json.dumps(dice_outcome, default=str)}\n"
            f"Open case file (durable, do not let these fade): {json.dumps(player_state.get('case_file', []), default=str)}\n"
            f"World facts (durable standing details - places, objects, history; reference these naturally, do not contradict them): {json.dumps(player_state.get('world_facts', {}), default=str)}\n"
            f"Recent scenes, verbatim, most recent scene LAST (reference material only, for checking "
            f"continuity - do not contradict or restart an action, capture, theft, injury, or death that "
            f"the final scene already depicts as complete): {json.dumps(player_state.get('recent_scenes_full', []), default=str)}\n"
            f"Earlier history, summarized and compressed (oldest first, chronological; each entry is a "
            f"short compression of a scene older than the verbatim ones above - treat these as established "
            f"fact about what happened, same as the verbatim scenes, just without exact wording; do not "
            f"contradict them): {json.dumps(player_state.get('earlier_scenes_summary', []), default=str)}\n"
            f"Current date (use exactly this, do not invent another): {player_state.get('current_date', 'unknown')}\n"
            f"Player state: {json.dumps(player_state, default=str)}"
        )
    else:
        has_scene_history = bool(player_state.get("recent_scenes_full")) or bool(player_state.get("earlier_scenes_summary"))

        if not has_scene_history:
            # No prior scene exists yet - this is the character's very
            # first turn (or a fresh session with nothing narrated so far).
            # The SAME SCENE / NEW BEAT branches below both assume a
            # "final scene in the narrative history" exists to continue or
            # pick up from - sending that instruction here was a real,
            # confirmed bug: told to continue a scene that doesn't exist,
            # the model produced confused, poorly-grounded choices on
            # exactly (and only) this turn, since every subsequent turn
            # genuinely has real history to anchor to and the same
            # instruction becomes accurate again immediately after. This
            # branch is the honest version: there is nothing to continue,
            # so start fresh from the action itself.
            continuity_mode = (
                "OPENING TURN: there is no prior scene yet - recent_scenes_full and "
                "earlier_scenes_summary are both empty because nothing has been narrated in this "
                "playthrough so far. Do not reference a 'final scene' or try to continue something "
                "that doesn't exist. Plan the scene_outline as a fresh start: establish where the "
                "character is and what's happening based on the action intent, current_date, "
                "location, and player_state alone.\n"
            )
        else:
            continuity_mode = (
                "SAME SCENE (no time skip this turn): the player's action below is their next move "
                "within the FINAL scene in the narrative history below (the one at the very end, after "
                "the last '---' separator), not a jump to a new moment. The people, place, objects, and "
                "situation already established in that final scene are still exactly as they were when it "
                "left off - nothing has reset. Plan the scene_outline as the direct next beat of that same "
                "scene (what the player does right now, and what happens in response, in the same place, "
                "same conversation, same few minutes), then let the scene continue or close from there. Do "
                "not invent a different scenario, skip ahead to an unrelated moment, or drop the specific "
                "people/objects/details that final scene named. The earlier scenes before it are broader "
                "recent context, not the scene to continue directly.\n"
                if not time_skip else
                "NEW BEAT (time has advanced this turn per the time-skip info below): the final scene in "
                "the narrative history has ALREADY ENDED - real time has passed since it, per the time-skip "
                "amount below. If that final scene was something naturally short and self-contained (a "
                "phone call, a single conversation, a meeting, a specific brief errand or encounter), it is "
                "over: do not continue it as if it is still happening. Summarize how it concluded in a "
                "sentence if that matters, then move on to what the character is doing now, at the new "
                "point in time. Only an activity that is genuinely still ongoing at this new time by its "
                "own nature (a job the character still holds, a relationship, a long-term project) should "
                "carry forward - and even then, as its current state now, not as a continuous scene "
                "resuming exactly where it left off. Honor anything the final scene left genuinely "
                "unresolved (a decision made, a promise given), but do not mistake an in-progress SCENE for "
                "an in-progress SITUATION - the scene is done; only the situation persists.\n"
            )

        user_content = (
            f"{_preset_instruction(player_state)}\n"
            "Plan the outcome of this action and return JSON matching the schema in your instructions.\n"
            f"Action intent: {action_intent}\n"
            f"Target word count for the writer to hit: {target_words} words. scene_outline should contain "
            f"roughly the right AMOUNT of content for this length - enough that the writer isn't forced "
            f"to pad or invent beats you didn't specify, but not so many distinct beats that hitting this "
            f"length would require rushing through or dropping things you listed. As a rough guide, a "
            f"scene at this length can comfortably hold one central beat with a few supporting details, "
            f"not several separate developments each needing their own space.\n"
            f"Choice granularity for this turn, if choices is non-empty: {choice_granularity}\n"
            f"{continuity_mode}"
            f"{override_skipped_note}"
            f"{time_skip_note}"
            f"Dice outcome: {json.dumps(dice_outcome, default=str)}\n"
            f"Open case file (durable, do not let these fade): {json.dumps(player_state.get('case_file', []), default=str)}\n"
            f"World facts (durable standing details - places, objects, history; reference these naturally, do not contradict them): {json.dumps(player_state.get('world_facts', {}), default=str)}\n"
            f"Recent scenes, verbatim, most recent scene LAST (reference material only, for checking "
            f"continuity - the final scene is what the player's action above is directly responding to; "
            f"do not contradict or restart an action, capture, theft, injury, or death that it already "
            f"depicts as complete): {json.dumps(player_state.get('recent_scenes_full', []), default=str)}\n"
            f"Earlier history, summarized and compressed (oldest first, chronological; each entry is a "
            f"short compression of a scene older than the verbatim ones above - treat these as established "
            f"fact about what happened, same as the verbatim scenes, just without exact wording; do not "
            f"contradict them): {json.dumps(player_state.get('earlier_scenes_summary', []), default=str)}\n"
            f"Current date (use exactly this, do not invent another): {player_state.get('current_date', 'unknown')}\n"
            f"Player state: {json.dumps(player_state, default=str)}"
        )

    try:
        character_age = int(player_state.get("age", 0))
    except (TypeError, ValueError):
        character_age = 0

    effective_system_prompt = PLANNING_SYSTEM_PROMPT
    if superuser_command:
        effective_system_prompt += SUPERUSER_COMMANDS_BLOCK
    if character_age >= AGE_ARC_MIN_AGE:
        effective_system_prompt += AGE_ARC_BLOCK

    pending_speed_change = player_state.get("pending_speed_change")
    if pending_speed_change:
        effective_system_prompt += _pace_change_context_note(pending_speed_change)

    return {
        "user_content": user_content,
        "effective_system_prompt": effective_system_prompt,
        "target_words": target_words,
        "dice_outcome": dice_outcome,
        "player_state": player_state,
    }


def plan_scene(player_state: dict, action_intent: str, dice_outcome: dict, superuser_command: str | None = None, time_skip: dict | None = None, force_choices: bool = False) -> dict:
    """
    The planning pass: reads case_file, life_events, recent scenes,
    earlier-history summaries, era context, and dice_outcome, and decides
    EVERYTHING about this turn before any prose exists - scene_outline,
    era_constraints, choices, state_deltas, scene_summary. Runs on the
    "planning" call_type (Qwen3-235B - see PLANNING_MODEL_PRIMARY's
    comment for why this is a plain, non-reasoning model rather than a
    reasoning one, despite the task's synthesis-heavy shape). This call's
    output is binding: write_scene does not get to override or reinterpret
    anything decided here, only turn scene_outline into prose.

    force_choices: when app.py's consecutive_empty_choice_turns counter has
    hit its cap, this re-calls plan_scene with FORCE_CHOICES_BLOCK attached
    - the retry happens HERE, before any prose has been written, unlike the
    old design where a failed choices check meant discarding and re-doing
    work after the (already-streamed) prose existed. Cheaper and cleaner:
    a rejected plan costs one more planning call, not a wasted prose call.
    """
    ctx = _build_planning_context(player_state, action_intent, dice_outcome, superuser_command, time_skip)

    effective_system_prompt = ctx["effective_system_prompt"]
    if force_choices:
        effective_system_prompt += FORCE_CHOICES_BLOCK

    # Diagnostic only - prints to the server log (visible in PythonAnywhere's
    # error log) whenever a pace-change note was included in this call, so
    # it's possible to directly confirm what's happening without needing
    # OpenRouter's dashboard. The pace-change note itself is now purely
    # informational (see _pace_change_context_note) - the actual
    # enforcement is Python-side in app.py's _apply_sim_speed_floor, so
    # what matters here is confirming the note appears and later cross-
    # checking against app.py's own logging of same_scene_streak /
    # force_ignore to see whether Python's forcing mechanism is firing
    # when expected.
    if "recently changed their ongoing pace" in effective_system_prompt:
        print(f"[pace-change diagnostic] Note included this call. "
              f"pending_speed_change={player_state.get('pending_speed_change')!r} "
              f"same_scene_streak_since_pace_change={player_state.get('same_scene_streak_since_pace_change')!r}")

    messages = [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": ctx["user_content"]},
    ]

    # 4000 tokens - reduced from an earlier 7000. The larger ceiling was
    # sized for a reasoning model's internal reasoning trace (consumed
    # from the same token budget as the visible output on most providers);
    # now that planning runs on a plain, non-reasoning model, there is no
    # reasoning trace to budget for, and the actual JSON output (a short
    # scene_outline, a handful of era_constraints, choices, state_deltas)
    # is only a few hundred tokens in practice. 4000 keeps real headroom
    # for a verbose turn without paying for capacity nothing will use.
    # 0.35 - deliberately conservative. plan_scene decides plot facts,
    # continuity, era constraints, choices, and state_deltas, not prose -
    # this is "boring correctness" territory the same way classification
    # is, not a place where more creative variation is desirable. Higher
    # temperature here risks inconsistent state (a wrong stat delta, an
    # invented detail that contradicts case_file) rather than more
    # colorful output, since there's no prose being written in this call
    # at all - see write_scene for where temperature was raised instead,
    # for exactly the opposite reason.
    parsed = _call_model(messages, temperature=0.35, max_tokens=4000, call_type="planning")

    if not isinstance(parsed, dict):
        parsed = {}

    parsed.setdefault("scene_outline", "")
    parsed.setdefault("era_constraints", [])
    parsed.setdefault("choices", [])
    parsed.setdefault("state_deltas", {})
    parsed.setdefault("scene_summary", "")

    if not isinstance(parsed["scene_outline"], str):
        parsed["scene_outline"] = str(parsed["scene_outline"])
    if not isinstance(parsed["era_constraints"], list):
        parsed["era_constraints"] = [str(parsed["era_constraints"])]
    if not isinstance(parsed["choices"], list):
        parsed["choices"] = [str(parsed["choices"])]
    if not isinstance(parsed["state_deltas"], dict):
        parsed["state_deltas"] = {}
    if not isinstance(parsed["scene_summary"], str):
        parsed["scene_summary"] = str(parsed["scene_summary"])

    parsed["target_words"] = ctx["target_words"]

    if "recently changed their ongoing pace" in effective_system_prompt:
        print(f"[pace-change diagnostic] Resulting scene_outline: {parsed.get('scene_outline', '')[:300]!r}")
        print(f"[pace-change diagnostic] Resulting choices: {parsed.get('choices')!r}")

    return parsed


def truncate_scene_if_needed(narrative_text: str, target_words: int, overrun_tolerance: float = 1.3) -> str:
    """
    Server-side, deterministic safety net against a scene running long -
    added after real production evidence that prompt instructions telling
    the writer to treat the word-count target as a hard ceiling were not
    reliably followed (Qwen3.7-flash was consistently hitting the
    max_tokens limit regardless of how explicitly this was stated). Rather
    than keep tightening prompt wording indefinitely against a model that
    has already shown it won't reliably self-regulate, this makes the
    ceiling actually enforced in Python.

    Only triggers when the scene exceeds target_words by more than
    overrun_tolerance (30% over by default) - small natural variance is
    expected and fine; this is specifically for correcting a scene that
    ran substantially over, not for policing every scene down to an exact
    count. When it does trigger, truncates at the last complete sentence
    boundary at or before the target word count, never mid-sentence or
    mid-word - a slightly-short-of-target scene that ends cleanly is far
    better than a longer one that got cut off wherever max_tokens happened
    to land, which is what was happening before this existed.
    """
    words = narrative_text.split()
    if len(words) <= target_words * overrun_tolerance:
        return narrative_text

    truncated_to_target = " ".join(words[:target_words])

    # Find the last sentence-ending punctuation at or before the target
    # cutoff, so the result always ends on a real sentence boundary rather
    # than wherever the word count happened to land.
    last_sentence_end = max(
        truncated_to_target.rfind(". "),
        truncated_to_target.rfind(".\n"),
        truncated_to_target.rfind("! "),
        truncated_to_target.rfind("!\n"),
        truncated_to_target.rfind("? "),
        truncated_to_target.rfind("?\n"),
    )

    if last_sentence_end == -1:
        # No sentence boundary found within the target at all (unusual -
        # would mean one extremely long run-on sentence) - fall back to
        # the word-count cutoff rather than returning nothing.
        return truncated_to_target.rstrip()

    return truncated_to_target[:last_sentence_end + 1].rstrip()


def write_scene(plan: dict):
    """
    Generator: the writing pass. Takes a plan (from plan_scene) and streams
    prose yielded chunk-by-chunk, for a caller (app.py) to forward to the
    frontend as it arrives. This model makes NO decisions - scene_outline,
    era_constraints, and target_words are already fully decided; its only
    job is prose craft. Runs on the "narration" call_type (Qwen3.7-flash
    primary). Falls back through NARRATION_MODEL_PRIMARY then
    NARRATION_MODEL_FALLBACK the same way _call_model does for the
    non-streaming path, but has to reimplement that loop here since
    _call_model assumes a single non-streaming JSON response.
    """
    era_constraints = plan.get("era_constraints") or []
    era_block = (
        "era_constraints for this scene: " + "; ".join(era_constraints) + "\n"
        if era_constraints else
        "era_constraints for this scene: none specified.\n"
    )

    user_content = (
        f"scene_outline (follow exactly): {plan.get('scene_outline', '')}\n"
        f"{era_block}"
        f"Target word count: {plan.get('target_words', 550)} words. Hit this number.\n"
    )

    messages = [
        {"role": "system", "content": WRITE_SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Per-model max_tokens, not one shared value: Ministral 14B 2512 has a
    # hard provider-side output cap of 4096 tokens (confirmed via its
    # OpenRouter model spec) - requesting more than that for THIS model
    # specifically risks the request being rejected or silently clamped,
    # the same class of mistake as the earlier reasoning-parameter bug
    # (assuming one model's constraints apply uniformly across a retry
    # tier). Qwen3.7-flash (the fallback) has no such cap in practice, so
    # it keeps a higher ceiling.
    MODEL_MAX_TOKENS = {
        "mistralai/ministral-14b-2512": 4096,
    }
    DEFAULT_WRITER_MAX_TOKENS = 5000

    models_to_try = [NARRATION_MODEL_PRIMARY, NARRATION_MODEL_FALLBACK]
    last_error: Exception | None = None

    for model in models_to_try:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                # 0.75, raised from 0.6 - this is the prose-writing call
                # specifically (plan_scene, which decides plot facts/
                # continuity/state_deltas, deliberately stays at a lower,
                # more conservative temperature - creativity there risks
                # inconsistent state, not "colorful" writing). A modest
                # bump here, not pushed further: enough to vary word
                # choice and sentence rhythm more than a flatter 0.6
                # tends to, without drifting into incoherence.
                temperature=0.75,
                top_p=0.95,
                frequency_penalty=0.1,
                presence_penalty=0.0,
                max_tokens=MODEL_MAX_TOKENS.get(model, DEFAULT_WRITER_MAX_TOKENS),
                stream=True,
                extra_body={"provider": {"sort": "throughput"}},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(f"All models in narration tier failed during streaming. Last error: {last_error}")


def narrate_outcome(player_state: dict, action_intent: str, dice_outcome: dict, superuser_command: str | None = None, time_skip: dict | None = None, force_choices: bool = False) -> dict:
    """
    Non-streaming compatibility wrapper: internally runs plan_scene, then
    write_scene to completion (collecting all chunks rather than yielding
    them), and returns the same {"narrative", "choices", "state_deltas",
    "scene_summary"} shape callers already expect. Existing callers
    (new_game's opening scene, the superuser-command branch) keep working
    unchanged; only app.py's main turn route needs to call plan_scene/
    write_scene directly to get the actual streaming benefit.
    """
    plan = plan_scene(player_state, action_intent, dice_outcome, superuser_command, time_skip, force_choices)

    narrative_chunks = []
    for chunk in write_scene(plan):
        narrative_chunks.append(chunk)
    narrative_text = "".join(narrative_chunks)
    narrative_text = truncate_scene_if_needed(narrative_text, plan.get("target_words", 550))

    return {
        "narrative": narrative_text,
        "choices": plan["choices"],
        "state_deltas": plan["state_deltas"],
        "scene_summary": plan.get("scene_summary", ""),
    }


CHARACTER_GEN_SYSTEM_PROMPT = """You generate a starting character for LifeSim, a grounded life simulator. Return ONLY valid JSON matching the schema below. Keep it realistic: no magic, no destiny, plausible economic and demographic details for the stated era and location. Stats are on a 0-20 scale (10 is an average human baseline). Relationships must use one of exactly these five tiers: Hostile, Cold, Neutral, Warm, Devoted, each paired with a short cause. Provide AT MOST 3 starting relationships.

background is always a piece of writing you produce, never a copy-paste of whatever the player typed. When the player supplies their own background material (a sentence, a few bullet points, a full paragraph), treat it as a brief or a pitch, not a draft to lightly polish: your job is to write the actual backstory that brief implies, with specific texture a bare description doesn't have on its own (concrete family detail, how they came to their current circumstances, one formative moment). If a player writes "grew up poor, father was a miner," that is the seed of a background, not the background itself - do not return it back close to verbatim. If the player gives no background material at all, invent one wholesale, grounded in whatever era/location/occupation is established. Facts the player stated explicitly are fixed and must not be contradicted; everything else in the background is yours to develop.

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

    parsed = _call_model(messages, temperature=0.2, call_type="utility")

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
"""


def stream_bio(character_data: dict):
    """
    Generator: streams the character-creation bio as plain text chunks,
    the same plain-text-not-JSON pattern write_scene uses for scene prose
    (response_format=json_object cannot produce valid partial JSON while
    streaming, so any call that benefits from live token-by-token display
    has to drop JSON mode and return plain text instead - this is why
    generate_bio's old {"bio": "..."} wrapper is gone here; the plain text
    IS the return value, nothing to unwrap). generate_bio below is now a
    thin wrapper that collects this generator to completion for callers
    that don't need streaming.
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

    models_to_try = [UTILITY_MODEL_PRIMARY, UTILITY_MODEL_FALLBACK]
    last_error: Exception | None = None

    for model in models_to_try:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.75,
                top_p=0.95,
                frequency_penalty=0.1,
                presence_penalty=0.0,
                max_tokens=600,
                stream=True,
                extra_body={"provider": {"sort": "throughput"}},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(f"All models in utility tier failed during bio streaming. Last error: {last_error}")


def generate_bio(character_data: dict) -> str:
    """
    Writes a short retrospective bio for the character-creation summary screen.
    Deliberately a separate call from narrate_outcome's opening-scene narration:
    reusing that narration here was the original bug (the summary screen just
    showed a copy of the actual first scene, verbatim, meaning the player read
    the same text twice). This generates genuinely different, non-scene prose.

    Non-streaming compatibility wrapper around stream_bio - collects the
    generator to completion. Callers that want live streaming (the
    /api/new-game/stream route) should call stream_bio directly instead.
    """
    try:
        chunks = []
        for chunk in stream_bio(character_data):
            chunks.append(chunk)
        bio = "".join(chunks).strip()
        return bio or (character_data.get("background") or "No background available.")
    except Exception:
        # Fall back to the raw background text rather than leaving the
        # summary screen blank if this call fails for any reason.
        return character_data.get("background") or "No background available."

    return parsed["bio"].strip()


# Real production evidence (not a guess): random-mode character generation
# consistently landed on the same narrow set of names (Arthur, Elias,
# Eleanor, Dorothy...), the same era bracket (roughly 1890-1950), and
# similar backgrounds, call after call - despite temperature already
# being 0.95, the highest in this file. Temperature affects sampling
# variance within whatever distribution the model leans toward; it does
# NOT reliably break a strong learned bias toward a specific "grounded
# period character" archetype, which is exactly what LLMs are documented
# to default to absent a concrete, externally-forced reason to do
# otherwise. "Surprise me with era, location, and background" gives the
# model nothing concrete to diverge FROM - it's asking for variety
# without supplying any actual randomness.
#
# The fix: generate the randomness in Python (genuinely random, not
# hoping the model samples differently) and hand the model a specific,
# concrete combination to build from every call. This doesn't dictate the
# character - the model still invents everything else - it just breaks
# the "same defaults every time" failure mode by giving it real varying
# material to react to instead of an open-ended prompt it keeps
# resolving the same way.
_ERA_POOL = [
    (1880, 1900), (1900, 1920), (1920, 1940), (1940, 1960),
    (1960, 1980), (1980, 2000), (2000, 2015), (2015, 2026),
]

_REGION_POOL = [
    "the Northeastern United States", "the Midwestern United States",
    "the Southern United States", "the Western United States",
    "the United Kingdom", "Western Europe", "Eastern Europe",
    "East Asia", "Southeast Asia", "South Asia", "Latin America",
    "the Caribbean", "Sub-Saharan Africa", "the Middle East",
    "Australia or New Zealand", "Canada",
]

_LIFE_SITUATION_POOL = [
    "someone from a working-class background",
    "someone from a wealthy or upper-class background",
    "an immigrant or first-generation resident",
    "someone from a rural or small-town background",
    "someone from a major city",
    "someone in a skilled trade or craft",
    "someone with an unconventional or nontraditional path",
    "someone from a military family",
    "someone from a religious or tightly-knit community",
    "someone estranged from most of their family",
]


def _random_character_seed() -> str:
    """
    Builds one concrete, randomized instruction line for random-mode
    character generation - see the comment above this function for why
    this exists. A fresh random combination every call, genuinely random
    (Python's random module), not left to the model to vary on its own.
    """
    era_start, era_end = random.choice(_ERA_POOL)
    region = random.choice(_REGION_POOL)
    situation = random.choice(_LIFE_SITUATION_POOL)
    return (
        f"For this character specifically: set it somewhere between {era_start} and {era_end}, "
        f"in or connected to {region}, and make them {situation}. Use this as real, specific "
        f"grounding - a genuine name, occupation, and background that actually fits this era, "
        f"region, and situation, not a generic one. Do not default to early-20th-century America "
        f"or a stock \"old-fashioned\" name regardless of what this combination calls for."
    )


def generate_character(custom_prompt: str | None = None) -> dict:
    """
    Generates a full starting character (background, era/location, age, stats,
    up to 3 relationships). If custom_prompt is given, treat it as creative
    constraints; otherwise generate something fully random and grounded.
    """
    instruction = (
        "Generate a starting character using the player's notes below as raw creative material, not "
        "literal text to copy. The player may give you anything from a single word or fact to a full "
        "paragraph - your job is to write a real, developed background from it, the same way a writer "
        "takes a one-line pitch and turns it into an actual character. Never just restate or lightly "
        "reword what the player wrote back as the background field; expand it into something with "
        "specific, grounded detail (family, upbringing, how they ended up where they are, a formative "
        "event) that is consistent with what they gave you but goes meaningfully further than it. Any "
        "concrete fact the player states (a name, an era, an occupation, a relationship, a stat) is a "
        "hard constraint and must be honored exactly; anything left unstated or only gestured at is "
        "yours to invent and develop.\n\n"
        f"Player's notes: {custom_prompt}"
        if custom_prompt
        else (
            "Generate a random, grounded starting character.\n\n" + _random_character_seed()
        )
    )

    messages = [
        {"role": "system", "content": CHARACTER_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    parsed = _call_model(messages, temperature=0.95, call_type="utility")

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


BACKGROUND_EXPAND_SYSTEM_PROMPT = """You expand a short piece of background material the player wrote for their LifeSim character into a real, developed backstory. The player's own text is a brief or a pitch, not a draft to lightly polish - do not restate or lightly reword it back. Write the actual backstory that brief implies: specific, grounded detail (family, upbringing, how they came to their current circumstances, one formative event) that goes meaningfully further than what they gave you, while staying consistent with it.

Any other character fields given below (name, age, location, occupation, era) are fixed facts already decided - the background must be consistent with them, not contradict or reinvent them. 2-4 sentences. Plain prose, no headers or formatting.

Return ONLY valid JSON, no markdown, in exactly this shape: {"background": "the expanded background"}
"""


def expand_background(player_text: str, other_fields: dict) -> str:
    """
    Expands a player-typed background into a real backstory, without
    touching or inventing any other character field - unlike
    generate_character, which builds a whole character from scratch, this
    is for the custom-character form specifically: the player fills in
    name/age/location/etc. manually and wants only the background field
    developed from what they typed, not overridden by a full AI-generated
    character. other_fields is passed as context so the expansion stays
    consistent with whatever else the player already specified (their
    stated occupation, era, location), not so those fields get rewritten.
    """
    if not player_text or not player_text.strip():
        return player_text

    context_lines = "\n".join(
        f"{key}: {value}" for key, value in other_fields.items() if value
    )

    messages = [
        {"role": "system", "content": BACKGROUND_EXPAND_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Player's background notes: {player_text}\n\n"
                f"Other fixed character facts:\n{context_lines or '(none given)'}"
            ),
        },
    ]

    parsed = _call_model(messages, temperature=0.75, call_type="utility")

    if not isinstance(parsed, dict) or not parsed.get("background"):
        # Fall back to the player's own text rather than losing it if this
        # call fails - worse than not expanding it, but never worse than
        # what they already had.
        return player_text

    return str(parsed["background"]).strip()
