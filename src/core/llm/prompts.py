import inspect

NPC_CONTEXT_BASE_QUEST_PROMPT = inspect.cleandoc("""
    Embody the following NPC completely that has the main TASK to give a QUEST to the main character. 
    Every line of the quest dialogue must authentically reflect their profile, mannerisms, and background. 
""")

NPC_CONTEXT_BASE_DIALOGUE_PROMPT = inspect.cleandoc("""
    Embody the following NPC completely. Every line of dialogue must authentically reflect their profile, mannerisms, and background. 
""")


NPC_FIELDS_PROMPT = inspect.cleandoc("""
    FIELD GUIDANCE:
    - Personality: Defines the NPC's emotional state, attitude, and moral compass.
    - Context: The NPC's current situation, objectives, and immediate environment.
    - Relationship: Dictates the initial level of trust, warmth, or hostility toward the main character.
    - Language / Dialect: Sets the vocabulary, tone, slang, or structural quirks of their speech.
    - Recent Events: Immediate past occurrences that should influence their current mood or focus.

    TALKATIVENESS GUIDE:
    Determines output length and verbosity (HOW MUCH they speak, not WHAT they say):
    - Very terse: Short, blunt sentences. Only essential words.
    - Reserved: Brief responses with minimal embellishment.
    - Balanced: Standard conversational length with moderate detail.
    - Talkative: Elaborates willingly, adding context, minor asides, or remarks.
    - Very talkative: Verbose and rambling; prone to tangents and extra detail.
    
    NPC FIELDS:
    - Name: {name},
    - Age: {age},
    - Personality: {personality},
    - Context: {context},
    - Talkativeness: {talkativeness},
    - Main Character Relation: {main_character_relation},
""")

DIALOGUE_BASE_PROMPT = inspect.cleandoc("""
    TASK:
    Write the dialogue line(s) this NPC would say to the main character.
""")


QUEST_BASE_PROMPT = inspect.cleandoc("""
    The NPC MUST use this conversation to offer and assign the specified Quest to the player.
    
    QUEST ASSIGNMENT MANDATE:
    Only use the given informations in the quest description, objective and more infos:
""")

QUEST_CHOICE_PROMPT = inspect.cleandoc("""
    - Beyond the {number_of_options} additional dialogue options, include explicitly 1 "accept" option and 1 "refuse" option in the \"player_options\" field. Both choices must directly address accepting or declining the quest's objective FROM THE MAIN CHARACTER'S POINT OF VIEW.
""")

DIALOGUE_OPTIONS_PROMPT = inspect.cleandoc("""
    - You MUST generate EXACTLY {number_of_options} additional dialogue options allowing the player to ask for details or context. These options MUST go into the \"dialogue_options\" field. NEVER allude at a possible acceptance or refusal of the quest when giving options. 
""")

ROLE_PROMPT = inspect.cleandoc("""
    You are a narrative designer generating dialogue for NPCs (non-player characters) in a videogame.
""")

WORLD_CONTEXT_PROMPT = inspect.cleandoc("""
    Use the following worldbuilding directives to shape the tone, dialogue, and atmospheric details of all generated content.
	
    WORLD CONTEXT:
    - Environment: {environment}
    - Epoch: {epoch}
    - Current Situation: {world_state}
""")

MAIN_CHARACTER_PROMPT = inspect.cleandoc("""
    The NPC will interact with the main character with the following description: 
    {main_character_description}
""")

DIALOGUE_HISTORY_PROMPT = inspect.cleandoc("""
    These are the main events of the current conversation between you and the main character:
    {dialogue_history}
""")

DIALOGUE_RULES_PROMPT = inspect.cleandoc("""

    TASK RULES:
""")

GENERAL_RULES_PROMPT = inspect.cleandoc("""
    GENERAL RULES:
    - If the NPC requires to give a Quest to the main character, it must use the given quest.
    - DO NOT make up any unexistent information and stick to the given informations only.
    - Stay consistent with the WORLD CONTEXT, the NPC's personality, and the overall tone of the setting.
    - Write dialogue in a natural, spoken style appropriate to the NPC's personality and the epoch.
    - Do not break the fourth wall or reference being an AI.
    - Respond ONLY with a valid JSON object matching the schema provided by the user, with no additional text, explanation, or markdown formatting.
""")

FAIRNESS_BASE_RULES_PROMPT = inspect.cleandoc("""
    - Avoid stereotypes related to the NPC's gender, ethnicity, nationality, or social background.
    - Do not associate negative traits (criminality, ignorance, aggression) with specific groups in a gratuitous manner or without justification in the narrative context.
""")