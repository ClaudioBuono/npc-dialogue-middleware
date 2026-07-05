from core.contexts import *
from enum import Enum

# Semantic constraints for Game Context (TODO: Move inside CONFIG).
MIN_EPOCH_LENGTH = 2
MAX_EPOCH_LENGTH = 100
MAX_ENVIRONMENT_LENGTH = 2000
MAX_PLOT_LENGTH = 5000
MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH = 2000

# Semantic constraints for NPC Context — (TODO: Move inside CONFIG).
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100
MIN_AGE = 0
MAX_AGE = 10_000
MAX_PERSONALITY_LENGTH = 1000
MAX_CONTEXT_LENGTH = 1000
MAX_RELATION_LENGTH = 500
MAX_RECENT_PLOT_LENGTH = 2000
MAX_VISUAL_DESCRIPTION_LENGTH = 2000
MAX_BACKSTORY_LENGTH = 5000
MAX_LANGUAGE_ENTRIES = 10
MAX_QUEST_NAME_LENGTH = 150
MAX_QUEST_OBJECTIVE_LENGTH = 500
MAX_QUEST_DESCRIPTION_LENGTH = 3000
MAX_QUEST_LOCATION_LENGTH = 200
MAX_QUEST_REWARD_LENGTH = 300
MAX_KEY_DIALOGUE_LENGTH = 1000


class ValidationErrorCode(Enum):
    """Semantic error codes for preprocessing validation failures."""

    EMPTY_FIELD = "empty_field"
    FIELD_TOO_LONG = "field_too_long"
    FIELD_TOO_SHORT = "field_too_short"
    INVALID_VALUE = "invalid_value"


class PreProcessingError(Exception):
    """Raised when semantic validation of an incoming model fails.

    Carries a machine-readable error code and a list of human-readable
    field-level error messages, so the API layer can translate this
    into an appropriate HTTP response without needing domain knowledge.
    """

    def __init__(self, code: ValidationErrorCode, errors: list[str]):
        self.code = code
        self.errors = errors
        super().__init__(f"[{code.value}] " + "; ".join(errors))

def validate_game_context(context: GameContext) -> GameContext:
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

    plot = context.plot.strip() if context.plot else None
    if plot is not None and len(plot) > MAX_PLOT_LENGTH:
        errors.append(f"Field 'plot' must not exceed {MAX_PLOT_LENGTH} characters.")

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
        raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=errors)

    return GameContext(
        epoch=epoch,
        environment=environment,
        plot=plot,
        main_character_description=main_character_description,
    )

def validate_npc_context(context: NPCContext) -> NPCContext:
    """Validate and normalize an NPCContext instance.

    Performs semantic validation beyond what Pydantic already checks at
    the type level (e.g. length constraints, blank-string checks, age
    bounds), including recursive validation of the nested `intent` field
    when present. Returns a normalized copy of the context.

    Args:
        context: The NPCContext instance to validate, already parsed and
            type-checked by Pydantic at the API boundary.

    Returns:
        A new NPCContext instance with normalized (stripped) string fields
        and a validated/normalized `intent`, if present.

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

    # `talkativeness` is an Enum: Pydantic already guarantees it's one of
    # the valid members at the type level, so no further checks are needed here.

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

    # --- Nested validation: intent (QuestDialogue | KeyDialogue | None) ---

    normalized_intent = None
    if isinstance(context.intent, QuestDialogue):
        normalized_intent, intent_errors = _validate_quest_dialogue(context.intent)
        errors.extend(intent_errors)
    elif isinstance(context.intent, KeyDialogue):
        normalized_intent, intent_errors = _validate_key_dialogue(context.intent)
        errors.extend(intent_errors)
    # context.intent is None -> normalized_intent stays None, no error.

    if errors:
        raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=errors)

    return NPCContext(
        name=name,
        age=context.age,
        personality=personality,
        context=npc_context_field,
        intent=normalized_intent,
        talkativeness=context.talkativeness,
        main_character_relation=main_character_relation,
        recent_plot=recent_plot,
        visual_description=visual_description,
        backstory=backstory,
        language=language,
    )

# --- PRIVATE METHODS ---

def _validate_quest_dialogue(quest: QuestDialogue) -> tuple[QuestDialogue, list[str]]:
    """Validate and normalize a QuestDialogue instance.

    Args:
        quest: The QuestDialogue instance to validate.

    Returns:
        A tuple of (normalized QuestDialogue, list of error messages).
        The error list is empty if validation succeeded.
    """
    errors: list[str] = []

    name = quest.name.strip()
    if not name:
        errors.append("Field 'intent.name' cannot be empty or whitespace only.")
    elif len(name) > MAX_QUEST_NAME_LENGTH:
        errors.append(f"Field 'intent.name' must not exceed {MAX_QUEST_NAME_LENGTH} characters.")

    objective = quest.objective.strip()
    if not objective:
        errors.append("Field 'intent.objective' cannot be empty or whitespace only.")
    elif len(objective) > MAX_QUEST_OBJECTIVE_LENGTH:
        errors.append(f"Field 'intent.objective' must not exceed {MAX_QUEST_OBJECTIVE_LENGTH} characters.")

    description = quest.description.strip()
    if not description:
        errors.append("Field 'intent.description' cannot be empty or whitespace only.")
    elif len(description) > MAX_QUEST_DESCRIPTION_LENGTH:
        errors.append(f"Field 'intent.description' must not exceed {MAX_QUEST_DESCRIPTION_LENGTH} characters.")

    location = quest.location.strip()
    if not location:
        errors.append("Field 'intent.location' cannot be empty or whitespace only.")
    elif len(location) > MAX_QUEST_LOCATION_LENGTH:
        errors.append(f"Field 'intent.location' must not exceed {MAX_QUEST_LOCATION_LENGTH} characters.")

    reward = quest.reward.strip() if quest.reward else None
    if reward is not None and len(reward) > MAX_QUEST_REWARD_LENGTH:
        errors.append(f"Field 'intent.reward' must not exceed {MAX_QUEST_REWARD_LENGTH} characters.")

    normalized = QuestDialogue(
        name=name,
        objective=objective,
        description=description,
        location=location,
        reward=reward,
    )
    return normalized, errors


def _validate_key_dialogue(key_dialogue: KeyDialogue) -> tuple[KeyDialogue, list[str]]:
    """Validate and normalize a KeyDialogue instance.

    Args:
        key_dialogue: The KeyDialogue instance to validate.

    Returns:
        A tuple of (normalized KeyDialogue, list of error messages).
        The error list is empty if validation succeeded.
    """
    errors: list[str] = []

    must_use = key_dialogue.must_use.strip()
    if not must_use:
        errors.append("Field 'intent.must_use' cannot be empty or whitespace only.")
    elif len(must_use) > MAX_KEY_DIALOGUE_LENGTH:
        errors.append(f"Field 'intent.must_use' must not exceed {MAX_KEY_DIALOGUE_LENGTH} characters.")

    normalized = KeyDialogue(must_use=must_use)
    return normalized, errors
