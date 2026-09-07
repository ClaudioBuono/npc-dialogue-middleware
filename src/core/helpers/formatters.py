def format_dialogue_history(history: list[dict[str, str]], npc_name: str) -> str:
    """Format the raw dialogue history into a human-readable block for the prompt.

    Args:
        history: List of single-key dicts, e.g. [{'NPC': '...'}, {'Player': '...'}].
                 The key indicates the speaker ('NPC' or 'Player'), the value is
                 the line spoken or the label of the choice made.
        npc_name: Display name of the NPC, used in place of the generic 'NPC' key.

    Returns:
        A formatted multi-line string ready to be embedded in the prompt.
        Returns an empty string if history is empty.
    """
    if not history:
        return ""

    lines = []
    for turn in history:
        if not turn:
            continue
        speaker, content = next(iter(turn.items()))
        content = content.strip()

        if speaker == "NPC":
            lines.append(f'- {npc_name}: "{content}"')
        else:
            lines.append(f"- Player: {content}")
    

    return "\n".join(lines)