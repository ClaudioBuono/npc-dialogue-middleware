from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from core.orchestrator import Orchestrator
from core.types.contexts import GameContext, NPCContext
from api.schemas import ComposedDialogue, DialogueStreamRequest
from api.errors import GLOBAL_ERROR_RESPONSES
router = APIRouter(tags=["dialogue"])

@router.get("/health", summary="Health Check", description="Returns OK if the service is running.")
def health():
    return {"status": "ok"}

@router.post(
	"/set-game-context",
	summary="Set Global Game Context",
	description="Sets the global game world context including environment, epoch, and world state. This should be called when the player enters a new zone or a major world event occurs.",
	responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Context set successfully."}}
)
def set_game_context(game_context: GameContext):
	Orchestrator().set_game_context(
		game_context.environment,
		game_context.epoch,
		game_context.world_state,
		game_context.main_character_description,
	)

	return {"status": "ok"}

@router.post(
	"/generate-dialogue",
	response_model=ComposedDialogue,
	summary="Generate NPC Dialogue",
	description="Generates dialogue and available player responses based on the provided NPC context and current intent.",
	responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Dialogue generated successfully."}}
)
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

@router.post(
	"/start-dialogue-stream",
	response_class=StreamingResponse,
	summary="Starts NPC Dialogue using streaming mode",
	description="Streams the generated dialogue line by line to reduce perceived latency for the player, cleaning the dialogue history.",
	responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Stream of dialogue text."}}
)
def start_dialogue_stream(npc_context: NPCContext):
	if Orchestrator().game_context is None:
		raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Game context is not set.",
        )

	Orchestrator().dialogue_history.clear_dialogue_history()

	stream = Orchestrator().generate_dialogue_stream(
		npc_context.name,
		npc_context.age,
		npc_context.personality,
		npc_context.context,
		npc_context.talkativeness,
		npc_context.main_character_relation,
		npc_context.intent,
		None
	)

	return StreamingResponse(stream, media_type="text/plain")


@router.post(
    "/continue-dialogue-stream",
    response_class=StreamingResponse,
    summary="Continues the NPC Dialogue using streaming mode",
    description="Streams the generated dialogue line by line to reduce perceived latency for the player, without cleaning the dialogue history.",
    responses={
        **GLOBAL_ERROR_RESPONSES,
        200: {"description": "Stream of dialogue text."},
    },
)
def continue_dialogue_stream(request: DialogueStreamRequest):
	if Orchestrator().game_context is None:
		raise HTTPException(
			status_code=status.HTTP_409_CONFLICT,
			detail="Game context is not set.",
		)

	npc = request.npc_context

	stream = Orchestrator().generate_dialogue_stream(
		npc.name,
		npc.age,
		npc.personality,
		npc.context,
		npc.talkativeness,
		npc.main_character_relation,
		npc.intent,
		request.last_player_choice,
	)

	return StreamingResponse(stream, media_type="text/plain")