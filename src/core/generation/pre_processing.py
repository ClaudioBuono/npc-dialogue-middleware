import logging
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar
from core.configuration.thresholds import *
from core.types.contexts import GameContext, Dialogue, NPCContext, Quest
from core.tools.errors import PreProcessingError, ValidationErrorCode

logger = logging.getLogger(__name__)


# --- RESULT TYPES ---

# Generic intent type, so that IntentValidationResult[Quest]
# and IntentValidationResult[Dialogue] are both valid, without
# the need to make two very similar dataclasses.
TIntent = TypeVar("TIntent", bound=Dialogue, covariant=True)


@dataclass(frozen=True)
class _DialogueBaseFieldsResult:
    """Result of validating the fields shared by `Dialogue` and `Quest`.

    Attributes:
        must_use_expression: The normalized (stripped) `must_use_expression`,
            or None if it was missing or empty.
        more_info: The normalized (stripped) `more_info`, or None if it was
            missing or empty.
        errors: Validation error messages. Empty if validation succeeded.
    """

    must_use_expression: Optional[str]
    more_info: Optional[str]
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _IntentValidationResult(Generic[TIntent]):
    """Result of validating an intent (`Dialogue` or `Quest`).

    Generic over the intent type, so that `_validate_quest` returns a
    `_IntentValidationResult[Quest]` and `_validate_dialogue` returns a
    `_IntentValidationResult[Dialogue]`.

    Attributes:
        normalized: The normalized copy of the validated intent. If the intent
            was inconsistent or unknown, this is the original, unmodified object.
        errors: Validation error messages. Empty if validation succeeded.
    """

    normalized: TIntent
    errors: list[str] = field(default_factory=list)


def normalize_and_validate_game_context(context: GameContext) -> GameContext:
    """Validate and normalize a GameContext instance.

    Performs semantic validation (beyond what Pydantic already checks
    at the type level, e.g. length constraints and blank-string checks)
    and returns a normalized copy of the context (whitespace-stripped).

    Args:
        context: The GameContext instance to validate, already parsed
            and type-checked by Pydantic at the API boundary.

    Returns:
        A new GameContext instance with normalized (stripped) string fields.

    Raises:
        PreProcessingError: If any field fails semantic validation.
    """
    errors: list[str] = []

    epoch = context.epoch.strip()
    if not epoch:
        errors.append("Field 'epoch' cannot be empty or whitespace only.")
    elif len(epoch) < MIN_EPOCH_LENGTH:
        errors.append(f"Field 'epoch' must be at least {MIN_EPOCH_LENGTH} characters long.")
    elif len(epoch) > MAX_EPOCH_LENGTH:
        errors.append(f"Field 'epoch' must not exceed {MAX_EPOCH_LENGTH} characters.")

    environment = context.environment.strip()
    if not environment:
        errors.append("Field 'environment' cannot be empty or whitespace only.")
    elif len(environment) > MAX_ENVIRONMENT_LENGTH:
        errors.append(f"Field 'environment' must not exceed {MAX_ENVIRONMENT_LENGTH} characters.")

    world_state = context.world_state.strip()
    if not world_state:
        errors.append("Field 'world_state' cannot be empty or whitespace only.")
    elif len(world_state) > MAX_WORLD_STATE_LENGTH:
        errors.append(f"Field 'world_state' must not exceed {MAX_WORLD_STATE_LENGTH} characters.")

    main_character_description = (
        context.main_character_description.strip()
        if context.main_character_description
        else None
    )
    if (
        main_character_description is not None
        and len(main_character_description) > MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH
    ):
        errors.append(
            f"Field 'main_character_description' must not exceed "
            f"{MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH} characters."
        )

    if errors:
        logger.warning(f"GameContext validation failed with {len(errors)} errors: {errors}")
        raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=errors)

    logger.debug("GameContext validated successfully.")
    return GameContext(
        epoch=epoch,
        environment=environment,
        world_state=world_state,
        main_character_description=main_character_description,
    )


def normalize_and_validate_npc_context(context: NPCContext) -> NPCContext:
    """Validate and normalize an NPCContext instance.

    Performs semantic validation beyond what Pydantic already checks at
    the type level (e.g. length constraints, blank-string checks, age
    bounds), including recursive validation of the nested `intent` field
    (delegated to `_validate_quest` or `_validate_dialogue`, which return an
    `_IntentValidationResult`). Returns a normalized copy of the context.

    Args:
        context: The NPCContext instance to validate, already parsed and
            type-checked by Pydantic at the API boundary.

    Returns:
        A new NPCContext instance with normalized (stripped) string fields
        and a validated/normalized `intent`.

    Raises:
        PreProcessingError: If any field fails semantic validation.
    """
    errors: list[str] = []

    name = context.name.strip()
    if not name:
        errors.append("Field 'name' cannot be empty or whitespace only.")
    elif len(name) < MIN_NAME_LENGTH:
        errors.append(f"Field 'name' must be at least {MIN_NAME_LENGTH} characters long.")
    elif len(name) > MAX_NAME_LENGTH:
        errors.append(f"Field 'name' must not exceed {MAX_NAME_LENGTH} characters.")

    if not (MIN_AGE <= context.age <= MAX_AGE):
        errors.append(f"Field 'age' must be between {MIN_AGE} and {MAX_AGE}.")

    personality = context.personality.strip()
    if not personality:
        errors.append("Field 'personality' cannot be empty or whitespace only.")
    elif len(personality) > MAX_PERSONALITY_LENGTH:
        errors.append(f"Field 'personality' must not exceed {MAX_PERSONALITY_LENGTH} characters.")

    npc_context_field = context.context.strip()
    if not npc_context_field:
        errors.append("Field 'context' cannot be empty or whitespace only.")
    elif len(npc_context_field) > MAX_CONTEXT_LENGTH:
        errors.append(f"Field 'context' must not exceed {MAX_CONTEXT_LENGTH} characters.")

    main_character_relation = context.main_character_relation.strip()
    if not main_character_relation:
        errors.append("Field 'main_character_relation' cannot be empty or whitespace only.")
    elif len(main_character_relation) > MAX_RELATION_LENGTH:
        errors.append(f"Field 'main_character_relation' must not exceed {MAX_RELATION_LENGTH} characters.")

    # --- Optional fields ---

    recent_plot = context.recent_plot.strip() if context.recent_plot else None
    if recent_plot is not None and len(recent_plot) > MAX_RECENT_PLOT_LENGTH:
        errors.append(f"Field 'recent_plot' must not exceed {MAX_RECENT_PLOT_LENGTH} characters.")

    visual_description = context.visual_description.strip() if context.visual_description else None
    if visual_description is not None and len(visual_description) > MAX_VISUAL_DESCRIPTION_LENGTH:
        errors.append(f"Field 'visual_description' must not exceed {MAX_VISUAL_DESCRIPTION_LENGTH} characters.")

    backstory = context.backstory.strip() if context.backstory else None
    if backstory is not None and len(backstory) > MAX_BACKSTORY_LENGTH:
        errors.append(f"Field 'backstory' must not exceed {MAX_BACKSTORY_LENGTH} characters.")

    language: Optional[list[str]] = None
    if context.language is not None:
        normalized_languages = [lang.strip() for lang in context.language if lang and lang.strip()]
        if not normalized_languages:
            errors.append("Field 'language', if present, must contain at least one non-empty entry.")
        elif len(normalized_languages) > MAX_LANGUAGE_ENTRIES:
            errors.append(f"Field 'language' must not contain more than {MAX_LANGUAGE_ENTRIES} entries.")
        language = normalized_languages

    # --- Nested validation: intent (Quest | Dialogue) ---

    intent_result: _IntentValidationResult[Quest | Dialogue]

    if context.intent.type == "Quest":
        if not isinstance(context.intent, Quest):
            errors.append(
                "Field 'intent.type' is 'Quest' but the object is not a Quest instance."
            )
            intent_result = _IntentValidationResult(normalized=context.intent)
        else:
            intent_result = _validate_quest(context.intent)
    elif context.intent.type == "Dialogue":
        if not isinstance(context.intent, Dialogue) or isinstance(context.intent, Quest):
            errors.append(
                "Field 'intent.type' is 'Dialogue' but the object is not a plain Dialogue instance."
            )
            intent_result = _IntentValidationResult(normalized=context.intent)
        else:
            intent_result = _validate_dialogue(context.intent)
    else:
        errors.append(f"Unknown value for field 'intent.type': '{context.intent.type}'.")
        intent_result = _IntentValidationResult(normalized=context.intent)

    errors.extend(intent_result.errors)

    if errors:
        logger.warning(f"NPCContext validation failed with {len(errors)} errors: {errors}")
        raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=errors)

    logger.debug("NPCContext validated successfully.")
    return NPCContext(
        name=name,
        age=context.age,
        personality=personality,
        context=npc_context_field,
        intent=intent_result.normalized,
        talkativeness=context.talkativeness,
        main_character_relation=main_character_relation,
        recent_plot=recent_plot,
        visual_description=visual_description,
        backstory=backstory,
        language=language,
    )


# --- PRIVATE METHODS ---

def _validate_dialogue_base_fields(dialogue: Dialogue) -> _DialogueBaseFieldsResult:
    """Validate and normalize the fields shared by `Dialogue` and `Quest`.

    Since `Quest` inherits from `Dialogue`, this helper is reused by both
    `_validate_quest` and `_validate_dialogue` to avoid duplicating the
    validation logic for `must_use_expression` and `more_info`.

    Args:
        dialogue: A Dialogue (or Quest, since it inherits from Dialogue) instance.

    Returns:
        A `_DialogueBaseFieldsResult` containing the normalized
        `must_use_expression`, the normalized `more_info`, and the list of
        error messages (empty if validation succeeded).
    """
    errors: list[str] = []

    must_use_expression = dialogue.must_use_expression.strip() if dialogue.must_use_expression else None
    if must_use_expression is not None and len(must_use_expression) > MAX_MUST_USE_EXPRESSION_LENGTH:
        errors.append(f"Field 'intent.must_use_expression' must not exceed {MAX_MUST_USE_EXPRESSION_LENGTH} characters.")

    more_info = dialogue.more_info.strip() if dialogue.more_info else None
    if more_info is not None and len(more_info) > MAX_MORE_INFO_LENGTH:
        errors.append(f"Field 'intent.more_info' must not exceed {MAX_MORE_INFO_LENGTH} characters.")

    number_of_options = dialogue.number_of_options
    if isinstance(number_of_options, bool) or not isinstance(number_of_options, int):
        errors.append("Field 'intent.number_of_options' must be an integer.")
    elif not (0 <= number_of_options <= MAX_NUMBER_OF_OPTIONS_LENGTH):
        errors.append(
            f"Field 'intent.number_of_options' must be in the range 0 and {MAX_NUMBER_OF_OPTIONS_LENGTH}."
        )

    return _DialogueBaseFieldsResult(
        must_use_expression=must_use_expression,
        more_info=more_info,
        errors=errors,
    )


def _validate_quest(quest: Quest) -> _IntentValidationResult[Quest]:
    """Validate and normalize a Quest instance.

    Args:
        quest: The Quest instance to validate.

    Returns:
        An `_IntentValidationResult[Quest]` containing the normalized Quest
        and the list of error messages. The error list is empty if
        validation succeeded.
    """
    base = _validate_dialogue_base_fields(quest)
    errors = base.errors

    name = quest.name.strip() if quest.name else None
    if name is not None and len(name) > MAX_QUEST_NAME_LENGTH:
        errors.append(f"Field 'intent.name' must not exceed {MAX_QUEST_NAME_LENGTH} characters.")

    objective = quest.objective.strip()
    if not objective:
        errors.append("Field 'intent.objective' cannot be empty or whitespace only.")
    elif len(objective) > MAX_QUEST_OBJECTIVE_LENGTH:
        errors.append(f"Field 'intent.objective' must not exceed {MAX_QUEST_OBJECTIVE_LENGTH} characters.")

    description = quest.description.strip() if quest.description else None
    if description is not None and len(description) > MAX_QUEST_DESCRIPTION_LENGTH:
        errors.append(f"Field 'intent.description' must not exceed {MAX_QUEST_DESCRIPTION_LENGTH} characters.")

    location = quest.location.strip() if quest.location else None
    if location is not None and len(location) > MAX_QUEST_LOCATION_LENGTH:
        errors.append(f"Field 'intent.location' must not exceed {MAX_QUEST_LOCATION_LENGTH} characters.")

    reward = quest.reward.strip() if quest.reward else None
    if reward is not None and len(reward) > MAX_QUEST_REWARD_LENGTH:
        errors.append(f"Field 'intent.reward' must not exceed {MAX_QUEST_REWARD_LENGTH} characters.")

    normalized = Quest(
        type="Quest",
        must_use_expression=base.must_use_expression,
        more_info=base.more_info,
        number_of_options=quest.number_of_options,
        objective=objective,
        name=name,
        description=description,
        location=location,
        reward=reward,
        has_choice=quest.has_choice,
    )
    return _IntentValidationResult(normalized=normalized, errors=errors)


def _validate_dialogue(dialogue: Dialogue) -> _IntentValidationResult[Dialogue]:
    """Validate and normalize a Dialogue instance.

    Args:
        dialogue: The Dialogue instance to validate.

    Returns:
        An `_IntentValidationResult[Dialogue]` containing the normalized
        Dialogue and the list of error messages. The error list is empty if
        validation succeeded.
    """
    base = _validate_dialogue_base_fields(dialogue)

    normalized = Dialogue(
        type="Dialogue",
        must_use_expression=base.must_use_expression,
        more_info=base.more_info,
        number_of_options=dialogue.number_of_options,
    )
    return _IntentValidationResult(normalized=normalized, errors=base.errors)
