from fastapi import APIRouter, HTTPException, status
from core.orchestrator import Orchestrator
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import ComposedDialogue

router = APIRouter(tags=["dialogue"])

@router.post("/set-game-context")
def set_game_context(game_context: GameContext):
	Orchestrator().set_game_context(
		game_context.environment,
		game_context.epoch,
		game_context.world_state,
		game_context.main_character_description,
	)

	return {"status": "ok"}

@router.post("/generate-dialogue", response_model=ComposedDialogue)
def generate_dialogue(npc_context: NPCContext):
	dialogue: ComposedDialogue = Orchestrator().generate_dialogue(
		npc_context.name,
		npc_context.age,
		npc_context.personality,
		npc_context.context,
		npc_context.talkativeness,
		npc_context.main_character_relation,
		npc_context.intent,
		None
	)

	if dialogue is None:
		raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dialogue generation was refused or game context is not set.",
        )

	return dialogue