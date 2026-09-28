from __future__ import annotations
from dataclasses import dataclass, field
from api.schemas import ComposedDialogue
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest


@dataclass
class EvalCase:
    """A single golden-dataset case for semantic judger evaluation.

    Attributes:
        id: Short unique identifier, used in reports and for locating
            specific cases (e.g. the consistency probe).
        game_context, npc_context, composed_dialogue: the real inputs
            passed to Judger.judge_dialogue.
        expected_categories: question ids (see Judger._build_questions)
            that the judge is expected to flag as failed for this case.
            Empty means "clean": no semantic category should fail.
        notes: free-text description for humans reading the report.
    """
    id: str
    game_context: GameContext
    npc_context: NPCContext
    composed_dialogue: ComposedDialogue
    expected_categories: set[str] = field(default_factory=set)
    notes: str = ""


@dataclass
class MetamorphicPair:
    """A (clean, mutated) pair isolating a single expected violation.

    clean_case must be judged with no issues in expected_category.
    mutated_case must be judged with exactly a failure in expected_category
    (other categories are not asserted on, to avoid coupling the test to
    unrelated wording of the mutation).
    """
    id: str
    clean_case: EvalCase
    mutated_case: EvalCase
    expected_category: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Shared fixtures for building contexts.
# ---------------------------------------------------------------------------

BASE_GAME_CONTEXT = GameContext(
    epoch="The Third Age",
    environment="A fortified trading town on the edge of the Greywood forest.",
    world_state="A truce between the town and the Greywood clans is one week old "
                 "and still fragile. No open conflict is currently happening.",
)


def make_npc_context(
    *,
    intent: Quest | Dialogue,
    name: str = "Maren the Blacksmith",
    age: int = 52,
    personality: str = "Gruff and no-nonsense, distrusts outsiders, rarely wastes "
                        "words, but fiercely loyal to those she calls friend.",
    context: str = "Working at her forge just after dusk, faint smell of coal smoke in the air.",
    talkativeness: str = "Balanced",
    main_character_relation: str = "Neutral acquaintance, cautiously respectful.",
    recent_plot: str | None = "A fragile truce with the Greywood clans was signed one week ago.",
    visual_description: str | None = None,
    backstory: str | None = None,
    language: list[str] | None = None,
) -> NPCContext:
    return NPCContext(
        intent=intent,
        name=name,
        age=age,
        personality=personality,
        context=context,
        talkativeness=talkativeness,
        main_character_relation=main_character_relation,
        recent_plot=recent_plot,
        visual_description=visual_description,
        backstory=backstory,
        language=language,
    )


def make_quest_intent(**overrides) -> Quest:
    defaults = dict(
        objective="Deliver a repaired sword to the town guard captain.",
        name="A Blade Reforged",
        description="The player has come to collect a sword Maren repaired overnight.",
        has_choice=False,
        must_use_expression=None,
    )
    defaults.update(overrides)
    return Quest(**defaults)


# ---------------------------------------------------------------------------
# Clean cases (no violation expected) — used to measure false positives.
# ---------------------------------------------------------------------------

_clean_pickup = EvalCase(
    id="clean_sword_pickup",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=make_quest_intent()),
    composed_dialogue=ComposedDialogue(
        intent=make_quest_intent(),
        dialogue=(
            "Here. Sharpened the edge myself, re-set the grip. Don't swing it "
            "like a fool and it'll outlast you."
        ),
    ),
    expected_categories=set(),
    notes="Terse, in-persona, factually grounded in context. Should pass everything.",
)

_clean_smalltalk = EvalCase(
    id="clean_smalltalk_weather",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=Dialogue()),
    composed_dialogue=ComposedDialogue(
        intent=Dialogue(),
        dialogue="Truce's a week old. Greywood folk still don't trust the gate guards. Can't blame them.",
    ),
    expected_categories=set(),
    notes="Plain Dialogue intent (no active quest, no has_choice field at all). "
          "Consistent with world_state, no invented entities, correct language, in-persona.",
)

# Reused by the consistency test: run this one N times and check stability.
_consistency_probe = EvalCase(
    id="consistency_probe",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=Dialogue()),
    composed_dialogue=ComposedDialogue(
        intent=Dialogue(),
        dialogue=(
            "Truce's holding. For now. Wouldn't wager coin on it lasting the season, "
            "but that's not really your concern, is it?"
        ),
    ),
    expected_categories=set(),
    notes=(
        "Deliberately a little more editorializing/borderline in tone than the "
        "other clean cases, to probe whether persona_consistency or faithfulness "
        "verdicts wobble across repeated runs at the configured temperature."
    ),
)


# ---------------------------------------------------------------------------
# Violation cases — one per semantic category, used to measure recall.
# ---------------------------------------------------------------------------

_violation_entity_check = EvalCase(
    id="violation_entity_anachronistic_item",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=make_quest_intent()),
    composed_dialogue=ComposedDialogue(
        intent=make_quest_intent(),
        dialogue=(
            "Careful on the road — bandits have an armored pickup truck now, "
            "mowing down guards with a mounted machine gun."
        ),
    ),
    expected_categories={"entity_check"},
    notes="Previous version used an invented-but-plausible fantasy faction name "
          "('Sunfire Legion'), which the entity_check question explicitly allows "
          "('present in the context or plausible within it') -- too weak a signal, "
          "and this case failed recall in a real run. Replaced with anachronistic "
          "items (pickup truck, machine gun) that cannot be plausible in any reading "
          "of a 'Third Age' fantasy setting, removing the ambiguity entirely.",
)

_violation_language = EvalCase(
    id="violation_language_mixed",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=make_quest_intent()),
    composed_dialogue=ComposedDialogue(
        intent=make_quest_intent(),
        dialogue="Here's your sword. Fait attention avec la lame, elle est tres tranchante.",
    ),
    expected_categories={"language"},
    notes="Second sentence switches to French; language question expects a single language.",
)

_violation_persona_consistency = EvalCase(
    id="violation_persona_out_of_character",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=Dialogue()),
    composed_dialogue=ComposedDialogue(
        intent=Dialogue(),
        dialogue=(
            "Oh, it is SO wonderful to see you again, truly! Let me tell you every "
            "little thing that has happened since we last spoke, I have missed our "
            "chats so very much, you dear, dear friend!"
        ),
    ),
    expected_categories={"persona_consistency"},
    notes="Maren's personality is gruff/reserved and main_character_relation is a "
          "cautious neutral acquaintance; this is effusive and warm, contradicting both.",
)

_violation_consistency = EvalCase(
    id="violation_world_state_contradiction",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=Dialogue()),
    composed_dialogue=ComposedDialogue(
        intent=Dialogue(),
        dialogue="Glad the truce has held five years now, not a single skirmish since.",
    ),
    expected_categories={"consistency", "faithfulness"},
    notes="world_state says the truce is one week old and fragile, not five years "
          "and settled. May legitimately trip both consistency and faithfulness; "
          "kept as a combined case rather than forcing an artificial split.",
)

_violation_fairness = EvalCase(
    id="violation_fairness_stereotype",
    game_context=BASE_GAME_CONTEXT,
    npc_context=make_npc_context(intent=Dialogue()),
    composed_dialogue=ComposedDialogue(
        intent=Dialogue(),
        dialogue=(
            "You can't trust folks from the coastal provinces with a contract, "
            "sneaky by nature, always has been, always will be."
        ),
    ),
    expected_categories={"fairness"},
    notes="Previous version targeted 'the Greywood folk', which overlaps with "
          "Maren's established 'distrusts outsiders' personality trait -- the "
          "fairness question explicitly excuses bias 'justified by the NPC's "
          "established persona', so the judge had a legitimate loophole and this "
          "case failed recall in a real run. Replaced with a group ('coastal "
          "provinces') that appears nowhere in her personality, backstory, or "
          "recent_plot, so no persona-based justification is available.",
)


GOLDEN_CASES: list[EvalCase] = [
    _clean_pickup,
    _clean_smalltalk,
    _consistency_probe,
    _violation_entity_check,
    _violation_language,
    _violation_persona_consistency,
    _violation_consistency,
    _violation_fairness,
]


# ---------------------------------------------------------------------------
# Metamorphic pairs: same clean base, one minimal targeted mutation each.
# ---------------------------------------------------------------------------

METAMORPHIC_PAIRS: list[MetamorphicPair] = [
    MetamorphicPair(
        id="entity_swap",
        clean_case=_clean_pickup,
        mutated_case=EvalCase(
            id="entity_swap_mutated",
            game_context=BASE_GAME_CONTEXT,
            npc_context=make_npc_context(intent=make_quest_intent()),
            composed_dialogue=ComposedDialogue(
                intent=make_quest_intent(),
                dialogue=(
                    "Here. Sharpened the edge myself with ore from the Voidforge Depths. "
                    "Don't swing it like a fool and it'll outlast you."
                ),
            ),
        ),
        expected_category="entity_check",
        notes="Only change: invented implausible place name inserted into an otherwise clean line.",
    ),
    MetamorphicPair(
        id="language_swap",
        clean_case=_clean_smalltalk,
        mutated_case=EvalCase(
            id="language_swap_mutated",
            game_context=BASE_GAME_CONTEXT,
            npc_context=make_npc_context(intent=Dialogue()),
            composed_dialogue=ComposedDialogue(
                intent=Dialogue(),
                dialogue="La tregua tiene, per ora. La gente del Greywood ancora non si fida delle guardie.",
            ),
        ),
        expected_category="language",
        notes="Only change: entire line translated to Italian, same content/meaning.",
    ),
    MetamorphicPair(
        id="persona_flip",
        clean_case=_clean_smalltalk,
        mutated_case=EvalCase(
            id="persona_flip_mutated",
            game_context=BASE_GAME_CONTEXT,
            npc_context=make_npc_context(intent=Dialogue()),
            composed_dialogue=ComposedDialogue(
                intent=Dialogue(),
                dialogue=(
                    "Oh, the truce! Isn't it just marvelous news? I could talk about it "
                    "all day, honestly, it's simply the best thing to happen around here!"
                ),
            ),
        ),
        expected_category="persona_consistency",
        notes="Only change: same topic, tone flipped from terse/gruff to effusive/talkative.",
    ),
]