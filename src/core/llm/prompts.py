import inspect

NPC_CONTEXT_BASE_PROMPT = inspect.cleandoc("""
    Embody the following NPC completely when generating dialogue. Every line must
    sound like something this specific character would say — reflect their
    personality, tone, and mannerisms consistently, not just their situation.

    Talkativeness: how inclined this NPC is to speak.
    Use it to determine the length and verbosity of the dialogue:
    - Very terse. Short, blunt sentences. Says only what's strictly necessary.
    - Reserved. Brief responses, few embellishments.
    - Balanced. Normal conversational length, some detail or color.
    - Talkative. Elaborates, adds context, asides, or small remarks.
    - Very talkative. Rambling, verbose, prone to tangents or extra detail.

    The talkativeness value affects HOW MUCH the NPC says, not WHAT they say — the
    core content and intent of the dialogue must remain unchanged regardless of
    the value.
""")

DIALOGUE_BASE_PROMPT = inspect.cleandoc("""
    Write the dialogue line(s) this NPC would say to the main character.
    Use the following details to shape who they are and how they speak:
""")

QUEST_BASE_PROMPT = inspect.cleandoc("""
    Make sure this NPC gives the following quest to the main character during
    the dialogue. Only use the given informations in the quest description, objective and more infos:
""")

QUEST_CHOICE_PROMPT = inspect.cleandoc("""
    Generate accept and refuse options for the dialogue, these are BEYOND the additional dialogue options.
    They must refer to the quest objective
""")

DIALOGUE_OPTIONS_PROMPT = inspect.cleandoc("""
    Generate EXACTLY {number_of_options} dialogue options as more infos to the dialogue, these MUST BE INCLUDED along accept/refuse options if included
""")

ROLE_PROMPT = inspect.cleandoc("""
    You are a narrative designer generating dialogue for NPCs (non-player characters) in a video game.
""")

WORLD_CONTEXT_PROMPT = inspect.cleandoc("""
    World context:
    - Setting: {environment}
    - Epoch: {epoch}
    - Plot: {world_state}
""")

MAIN_CHARACTER_PROMPT = inspect.cleandoc("""
    You are talking to the main character with this context:
    - Character Description: {main_character_description}
""")

DIALOGUE_HISTORY_PROMPT = inspect.cleandoc("""
    These are the main events of the current conversation between you and the player:
    {dialogue_history}
""")

GENERAL_RULES_PROMPT = inspect.cleandoc("""
    General rules:
    - If the NPC requires to give a Quest to the main character, it must use the given quest.
    - DO NOT make up any unexistent information and stick to the given informations only.
    - Stay consistent with the world context, the NPC's personality, and the overall tone of the setting.
    - Write dialogue in a natural, spoken style appropriate to the NPC's personality and the epoch.
    - Do not break the fourth wall or reference being an AI.
    - Respond ONLY with a valid JSON object matching the schema provided by the user, with no additional text, explanation, or markdown formatting.
""")

FAIRNESS_BASE_RULES_PROMPT = inspect.cleandoc("""
    - Avoid stereotypes related to the NPC's gender, ethnicity, nationality, or social background.
    - Do not associate negative traits (criminality, ignorance, aggression) with specific groups in a gratuitous manner or without justification in the narrative context.
""")