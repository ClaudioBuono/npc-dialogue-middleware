NPC_CONTEXT_BASE_PROMPT = """
        Embody the following character completely when generating dialogue.
    """

DIALOGUE_BASE_PROMPT = """
    Write the dialogue line(s) this NPC would say to the main character.
    Use the following details to shape who they are and how they speak:
"""

QUEST_BASE_PROMPT = """
    Make sure this NPC gives the following quest to the main character during
    the dialogue. Only use the given informations in the quest description, objective and more infos:
"""

QUEST_CHOICE_PROMPT = """Generate accept and refuse options for the dialogue.
                        They must refer to the quest objective"""

DIALOGUE_OPTIONS_PROMPT = """Generate more options for the dialogue, to have more informations beyond accept/refuse, generate the following number of """

ROLE_PROMPT = (
    "You are a narrative designer generating dialogue for NPCs (non-player "
    "characters) in a video game.\n"
)

WORLD_CONTEXT_PROMPT = (
    "World context:\n"
    "- Setting: {environment}\n"
    "- Epoch: {epoch}\n"
    "- Plot: {world_state}\n"
)

MAIN_CHARACTER_PROMPT = (
    "You are talking to the main character with this context:\n"
    "- Character Description: {main_character_description}\n"
)

GENERAL_RULES_PROMPT = (
    "General rules:\n"
    "- If the NPC requires to give a Quest to the main character, it must use the given quest."
    "- DO NOT make up any unexistent information and stick to the given informations only."
    "- Stay consistent with the world context, the NPC's personality, and the "
    "overall tone of the setting.\n"
    "- Write dialogue in a natural, spoken style appropriate to the NPC's "
    "personality and the epoch.\n"
    "- Do not break the fourth wall or reference being an AI.\n"
    "- Respond ONLY with a valid JSON object matching the schema provided by "
    "the user, with no additional text, explanation, or markdown formatting.\n"
)
