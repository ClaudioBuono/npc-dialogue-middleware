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
    - Name: {name}
    - Age: {age}
    - Personality: {personality}
    - Context: {context}
    - Talkativeness: {talkativeness}
    - Main Character Relation: {main_character_relation}
""")

DIALOGUE_BASE_PROMPT = inspect.cleandoc("""
    TASK:
    Write the dialogue line(s) this NPC would say to the main character.
""")


QUEST_BASE_PROMPT = inspect.cleandoc("""
    The NPC MUST use this conversation to offer and assign the specified Quest to the player.

    QUEST TO ASSIGN (MANDATORY CONTENT):
    This objective is a required fact, not optional flavor. The dialogue MUST explicitly
    communicate it to the player, phrased in the NPC's own voice/style. Do not omit it,
    generalize it away, or replace it with a vaguer version. Use only the informations below;
    do not add extra quest details beyond what is given:
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
    DIALOGUE HISTORY:
    These are the main events of the current conversation between you and the main character:
    {dialogue_history}
""")

DIALOGUE_RULES_PROMPT = inspect.cleandoc("""
    TASK RULES:
""")

GENERAL_RULES_PROMPT = inspect.cleandoc("""
    GENERAL RULES:
    - If a Quest is provided, its objective is MANDATORY content: the NPC's dialogue must explicitly convey it, never omit or water it down.
    - Do not invent factual information beyond what is explicitly given (see GROUNDING RULES for what counts as invented).
    - Stay consistent with the WORLD CONTEXT, the NPC's personality, and the overall tone of the setting.
    - Write dialogue in a natural, spoken style appropriate to the NPC's personality and the epoch.
    - Do not break the fourth wall or reference being an AI.
    - Respond ONLY with a valid JSON object matching the schema provided by the user, with no additional text, explanation, or markdown formatting.
""")

LANGUAGE_RULE_PROMPT = inspect.cleandoc("""
    OUTPUT LANGUAGE:
    All text values in the JSON (dialogues, descriptions, options) MUST be written entirely in {language}.
""")

FAIRNESS_BASE_RULES_PROMPT = inspect.cleandoc("""
    FAIRNESS RULES:
    - Avoid stereotypes related to the NPC's gender, ethnicity, nationality, or social background.
    - Do not associate negative traits (criminality, ignorance, aggression) with specific groups in a gratuitous manner or without justification in the narrative context.
""")

GROUNDING_RULES_PROMPT = inspect.cleandoc("""
    GROUNDING RULES:
    - Treat all information given above (NPC fields, World Context, Quest details, Recent Events) as the complete and only known facts about this character and situation.
    - You MAY freely invent minor stylistic and atmospheric details that do not add new facts: gestures, tone of voice, background sounds, physical actions, filler phrases consistent with the NPC's personality and dialect.
    - You MUST NOT invent new factual content: no new names of people or places, no new past events, no new relationships, no new causes or motivations, no new quest details beyond what was explicitly provided.
    - If the dialogue would naturally benefit from a specific detail that was not provided, keep the reference generic or vague rather than inventing specifics.
    - Never contradict any of the given fields.
    - This rule governs INVENTED content only. It does NOT permit omitting any mandatory content explicitly required elsewhere (e.g. the quest objective, required dialogue options). Grounding means not adding facts, never omitting required ones.
""")

FINAL_CHECK_PROMPT = inspect.cleandoc("""
    FINAL CHECK BEFORE ANSWERING (verify silently, then output only the JSON):
    1. If a quest was provided, does "dialogue" explicitly state its objective in full? It must.
    2. Does "dialogue_options" contain exactly the required number of neutral items, with no accept/refuse hints?
    3. If accept/refuse options were required, do they clearly reference the same objective?
    4. Is every fact used present in the fields provided above, with no invented names, places, or events?
    If any check fails, revise the content before producing the final output.
""")