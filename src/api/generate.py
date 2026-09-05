from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from api.handlers import _MIDDLEWARE_ERROR_STATUS_MAP
from core.StateManager import StateManager
from core.orchestrator import Orchestrator
from core.types.contexts import GameContext, NPCContext
from api.schemas import ComposedDialogue, DialogueStreamRequest
from api.errors import GLOBAL_ERROR_RESPONSES
from core.types.enums import MiddlewareState
from tools.errors import MiddlewareError, MiddlewareErrorCode
router = APIRouter(tags=["dialogue"])

@router.get("/health", summary="Health Check", description="Returns OK if the service is running.")
def health():
    return {"status": "ok"}

@router.get(
    "/status",
    summary="Middleware Status",
    description="Returns the current middleware state.",
    responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Middleware is idling."}}
)
def middleware_status():
    state_manager = StateManager()
    current_state = state_manager.state

    match current_state:
        case MiddlewareState.IDLE:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"state": current_state.value},
            )
        case MiddlewareState.STARTING:
            error_code = MiddlewareState.STARTING
            message = "The middleware is starting."
        case MiddlewareState.SETTING_CONTEXT:
            error_code = MiddlewareState.SETTING_CONTEXT
            message = "The middleware is setting the game context."
        case MiddlewareState.GENERATING:
            error_code = MiddlewareErrorCode.GENERATING
            message = "The middleware is generating the dialogue."
        case _:
            error_code = None

    http_status = _MIDDLEWARE_ERROR_STATUS_MAP.get(
        error_code, status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=http_status,
        content={
            "state": current_state.value,
            "error_code": error_code.value if error_code else None,
            "message": message
        },
    )

@router.post(
	"/set-game-context",
	summary="Set Global Game Context",
	description="Sets the global game world context including environment, epoch, and world state. This should be called when the player enters a new zone or a major world event occurs.",
	responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Context set successfully."}}
)
def set_game_context(game_context: GameContext):

	if not StateManager().is_in(MiddlewareState.IDLE):
			raise MiddlewareError(code=MiddlewareErrorCode.SETTING_CONTEXT, errors=["The middleware is busy setting the game context."])
      
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

	if Orchestrator().game_context is None:
			raise HTTPException(
				status_code=status.HTTP_409_CONFLICT,
				detail="Game context is not set.",
			)
       
	if not StateManager().is_in(MiddlewareState.IDLE):
		raise MiddlewareError(code=MiddlewareErrorCode.GENERATING, errors=["The middleware is busy generating."])
	  
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
    responses={**GLOBAL_ERROR_RESPONSES, 200: {"description": "Stream of dialogue text."}},
)
def start_dialogue_stream(npc_context: NPCContext):
    if Orchestrator().game_context is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Game context is not set.",
        )

    if not StateManager().is_in(MiddlewareState.IDLE):
        raise MiddlewareError(
            code=MiddlewareErrorCode.GENERATING,
            errors=["The middleware is busy generating."],
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
        None,
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